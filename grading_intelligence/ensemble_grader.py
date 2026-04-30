"""
Ensemble Grader — Multi-model cross-validated grading.

Sends grading prompts to multiple LLM providers via the handwriting-engine
consensus system, cross-validates scores, and returns confidence-weighted
results. Falls back gracefully to single-model when only one provider is
available.
"""

from __future__ import annotations

import concurrent.futures
import json
import logging
import re
import time
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

from grading_intelligence.structured_output import (
    ConfidenceLevel,
    Deduction,
    FreeResponseResult,
    GradingResult,
    ModelVote,
    QuestionResult,
    SectionResult,
    parse_legacy_grade_text,
)
from grading_intelligence.prompt_library import wrap_prompt_for_structured_output


@dataclass
class EnsembleConfig:
    """Configuration for ensemble grading."""
    strategy: str = "vote"  # "single", "best_of", "vote", "debate", "cascade"
    providers: list[str] | None = None  # None = auto-detect available
    confidence_threshold: float = 0.8
    structured_output: bool = True  # Use JSON output format
    fallback_to_legacy: bool = True  # Parse text if JSON fails
    max_score_divergence: float = 0.15  # Flag if models disagree by >15% of total
    cost_optimize: bool = False  # Enable cost-based routing of questions to providers
    fr_consistency_passes: int = 1  # Grade FR questions N times with same model; flag disagreements (>1 enables self-consistency)


class EnsembleGrader:
    """
    Multi-model grading orchestrator.

    Wraps the handwriting-engine consensus system for grading-specific
    workflows: prompt formatting, structured output parsing, score
    cross-validation, and audit trail generation.
    """

    def __init__(self, config: EnsembleConfig | None = None):
        self.config = config or EnsembleConfig()
        self._validate_providers()

    def _validate_providers(self):
        """Check which providers are actually available."""
        try:
            from handwriting_engine.providers import available_providers
            self._available = available_providers()
        except ImportError:
            # Handwriting engine not installed — fall back to the direct grading provider
            from grader.config import normalize_grading_provider, GRADING_PROVIDER
            self._available = [normalize_grading_provider(GRADING_PROVIDER)]

        if self.config.providers:
            # Filter to only available providers
            self.config.providers = [
                p for p in self.config.providers if p in self._available
            ]
            if not self.config.providers:
                self.config.providers = self._available[:1]  # At least one

    def grade_exam(
        self,
        page_images: list[dict],
        grading_prompt: str,
        max_long_side: int = 1568,
        jpeg_quality: int = 85,
        student_file: str = "",
        total_points: float = 0,
    ) -> GradingResult:
        """
        Grade an exam submission using ensemble or single-model grading.

        Args:
            page_images: List of dicts with 'path' keys
            grading_prompt: Formatted prompt from exam_prompt.py
            max_long_side: Max image dimension
            jpeg_quality: JPEG quality
            student_file: Student filename for audit
            total_points: Total possible points

        Returns:
            Structured GradingResult with full audit trail
        """
        start_time = time.time()

        # Decide strategy based on available providers
        strategy = self._effective_strategy()

        if strategy == "single":
            result = self._grade_single_model(
                page_images, grading_prompt, max_long_side, jpeg_quality
            )
        else:
            result = self._grade_ensemble(
                page_images, grading_prompt, max_long_side, jpeg_quality, strategy
            )

        # Populate metadata
        result.student_file = student_file
        result.grading_mode = "exam"
        result.consensus_strategy = strategy
        result.grading_duration_ms = int((time.time() - start_time) * 1000)

        if total_points and result.total_possible == 0:
            result.total_possible = total_points

        # FR self-consistency: re-grade FR questions N times, flag disagreements
        if self.config.fr_consistency_passes > 1 and result.free_response_results:
            result = self._apply_fr_consistency(
                result, page_images, grading_prompt, max_long_side, jpeg_quality
            )

        return result

    def grade_notebook(
        self,
        page_images: list[dict],
        rubric_prompt: str,
        answer_key: dict,
        survey_images: list[dict] | None = None,
        max_long_side: int = 1568,
        jpeg_quality: int = 85,
        student_file: str = "",
    ) -> GradingResult:
        """
        Grade a notebook submission using ensemble or single-model grading.

        Args:
            page_images: List of dicts with 'path' keys
            rubric_prompt: Formatted rubric prompt from rubric.py
            answer_key: The answer key dict
            survey_images: Optional low-res survey images
            max_long_side: Max image dimension
            jpeg_quality: JPEG quality
            student_file: Student filename for audit

        Returns:
            Structured GradingResult
        """
        start_time = time.time()
        strategy = self._effective_strategy()

        if strategy == "single":
            result = self._grade_single_model(
                page_images, rubric_prompt, max_long_side, jpeg_quality,
                mode="notebook",
            )
        else:
            result = self._grade_ensemble(
                page_images, rubric_prompt, max_long_side, jpeg_quality, strategy,
                mode="notebook",
            )

        result.student_file = student_file
        result.grading_mode = "notebook"
        result.assignment_title = answer_key.get("title", "")
        result.consensus_strategy = strategy
        result.grading_duration_ms = int((time.time() - start_time) * 1000)

        return result

    def _effective_strategy(self) -> str:
        """Determine the actual strategy to use based on available providers."""
        if self.config.strategy == "single":
            return "single"

        providers = self.config.providers or self._available
        if len(providers) < 2:
            return "single"

        return self.config.strategy

    def _apply_cost_routing(self, prompt: str, providers: list[str]) -> list[str]:
        """
        Apply cost-based routing: analyze prompt complexity and select the
        cheapest provider that can handle it accurately.

        Returns a (possibly reordered) provider list where the first provider
        is the cost-optimal choice.
        """
        from grading_intelligence.cost_optimizer import (
            classify_question_complexity,
            QuestionComplexity,
            route_to_model,
        )

        # Extract question metadata from the grading prompt to assess complexity
        prompt_lower = prompt.lower()
        complexity = QuestionComplexity.SIMPLE  # default

        # Check for handwriting-critical indicators
        handwriting_patterns = [
            "single letter", "single-letter", "fill in the letter",
            "circle one letter", "write the letter",
        ]
        if any(p in prompt_lower for p in handwriting_patterns):
            complexity = QuestionComplexity.HANDWRITING_CRITICAL

        # Check for complex/free-response indicators
        elif any(p in prompt_lower for p in [
            "explain", "describe the process", "free response",
            "show your work", "calculate", "discuss", "analyze",
            "compare and contrast", "evaluate",
        ]):
            complexity = QuestionComplexity.COMPLEX

        # Check for moderate indicators
        elif any(p in prompt_lower for p in [
            "short answer", "list the", "identify", "matching",
            "fill in the blank", "name the",
        ]):
            complexity = QuestionComplexity.MODERATE

        # Route based on complexity
        tier, recommended_model = route_to_model(complexity)
        logger.debug("[Cost Optimizer] Complexity=%s, Tier=%s, Model=%s", complexity.value, tier.value, recommended_model)

        # Map recommended model to provider name
        model_to_provider = {
            "gemini-2.5-flash": "gemini",
            "gemini-2.5-pro": "gemini",
            "gpt-4.1": "openai",
            "gpt-4.1-mini": "openai",
            "claude-sonnet-4-20250514": "claude",
            "claude-haiku-4-5-20251001": "claude",
        }
        preferred_provider = model_to_provider.get(recommended_model, "claude")

        # Reorder providers to put the preferred one first
        if preferred_provider in providers:
            reordered = [preferred_provider] + [p for p in providers if p != preferred_provider]
            return reordered

        # If preferred provider isn't available, fall back to original order
        logger.debug("[Cost Optimizer] Preferred provider '%s' not available, using %s", preferred_provider, providers[0])
        return providers

    def _grade_single_model(
        self,
        page_images: list[dict],
        prompt: str,
        max_long_side: int,
        jpeg_quality: int,
        mode: str = "exam",
    ) -> GradingResult:
        """Grade using the existing single-model pipeline (backward compatible)."""
        from grader.vision import (
            grade_exam_with_vision,
            grade_with_vision,
            new_usage_tracker,
        )
        from grader.image_optimizer import build_image_blocks, batch_pages_by_size

        tracker = new_usage_tracker()

        # Optionally wrap for structured output
        grading_prompt = prompt
        if self.config.structured_output:
            grading_prompt = wrap_prompt_for_structured_output(prompt, mode)

        if mode == "exam":
            raw_text = grade_exam_with_vision(
                page_images, grading_prompt,
                max_long_side=max_long_side,
                jpeg_quality=jpeg_quality,
                tracker=tracker,
            )
        else:
            # Notebook mode needs answer_key param — use raw prompt approach
            raw_text = grade_exam_with_vision(
                page_images, grading_prompt,
                max_long_side=max_long_side,
                jpeg_quality=jpeg_quality,
                tracker=tracker,
            )

        # Parse result
        result = self._parse_response(raw_text, mode)
        single_model_confidence = self._estimate_single_model_confidence(result)

        # Record model vote against the active single-provider configuration
        from grader.config import get_grading_model, normalize_grading_provider, GRADING_PROVIDER
        active_provider = normalize_grading_provider(GRADING_PROVIDER)
        result.model_votes.append(ModelVote(
            provider=active_provider,
            model_id=get_grading_model(active_provider),
            score=result.total_score,
            total=result.total_possible,
            raw_text=raw_text,
            confidence=single_model_confidence,
            tokens_used={
                "input_tokens": tracker["input_tokens"],
                "output_tokens": tracker["output_tokens"],
            },
        ))
        result.consensus_confidence = single_model_confidence
        result.total_input_tokens = tracker["input_tokens"]
        result.total_output_tokens = tracker["output_tokens"]
        result.total_api_calls = tracker["api_calls"]

        return result

    def _estimate_single_model_confidence(self, result: GradingResult) -> float:
        """
        Estimate overall confidence from question-level evidence for single-model runs.

        This keeps the review thresholds unchanged while allowing genuinely clean,
        well-read papers to clear them.
        """
        evidence = result.question_results or result.section_results or result.free_response_results
        if not evidence:
            return 0.70

        high_conf = 0
        reading_high = 0
        faint_or_illegible = 0
        for item in result.question_results:
            if item.confidence == ConfidenceLevel.HIGH:
                high_conf += 1
            if getattr(item, "reading_confidence", "HIGH") == "HIGH":
                reading_high += 1
            if getattr(item, "faint_content", False) or getattr(item, "illegible", False):
                faint_or_illegible += 1

        total = max(1, len(result.question_results))
        if not result.question_results:
            total = max(1, len(evidence))
            high_conf = sum(1 for item in evidence if item.confidence == ConfidenceLevel.HIGH)
            reading_high = high_conf

        high_ratio = high_conf / total
        reading_ratio = reading_high / total
        penalty = min(0.20, faint_or_illegible * 0.04)

        score = 0.52 + (0.28 * high_ratio) + (0.16 * reading_ratio) - penalty
        if result.total_possible > 0 and result.total_score in (0, result.total_possible):
            score = min(score, 0.69)
        return max(0.35, min(0.93, round(score, 3)))

    def _grade_ensemble(
        self,
        page_images: list[dict],
        prompt: str,
        max_long_side: int,
        jpeg_quality: int,
        strategy: str,
        mode: str = "exam",
    ) -> GradingResult:
        """
        Grade using multi-model ensemble via handwriting-engine providers.

        Each provider independently grades the submission, then results
        are cross-validated and merged.
        """
        from grader.image_optimizer import build_image_blocks, batch_pages_by_size
        from handwriting_engine.providers import get_provider
        from grader.vision import GRADING_SYSTEM_PROMPT

        providers = self.config.providers or self._available[:3]

        # Cost-based routing: override provider selection when cost_optimize is enabled
        if self.config.cost_optimize:
            providers = self._apply_cost_routing(prompt, providers)

        grading_prompt = prompt
        if self.config.structured_output:
            grading_prompt = wrap_prompt_for_structured_output(prompt, mode)

        # Build image blocks from ALL batches (not just the first)
        batches = batch_pages_by_size(page_images, max_long_side, jpeg_quality)
        image_blocks: list[dict] = []
        for batch in batches:
            image_blocks.extend(build_image_blocks(batch, max_long_side, jpeg_quality))

        # Grade with each provider in parallel
        model_votes: list[ModelVote] = []
        provider_results: list[GradingResult] = []

        def _grade_with_provider(provider_name: str) -> tuple[str, ModelVote, GradingResult | None]:
            """Run a single provider grading call. Returns (name, vote, parsed_or_None)."""
            try:
                provider = get_provider(provider_name)
                start = time.time()

                raw_text = provider.read_batch(
                    image_blocks,
                    grading_prompt,
                    system_prompt=GRADING_SYSTEM_PROMPT,
                    max_tokens=8192,
                )

                latency = int((time.time() - start) * 1000)
                parsed = self._parse_response(raw_text, mode)

                vote = ModelVote(
                    provider=provider_name,
                    model_id=getattr(provider, 'model', provider_name),
                    score=parsed.total_score,
                    total=parsed.total_possible,
                    raw_text=raw_text,
                    confidence=0.7,
                    tokens_used=provider.usage if hasattr(provider, 'usage') else {},
                    latency_ms=latency,
                )
                return provider_name, vote, parsed

            except (
                json.JSONDecodeError, ValueError, KeyError,
                ConnectionError, TimeoutError, OSError,
                RuntimeError,  # Covers provider-specific API errors that wrap stdlib
            ) as e:
                logger.warning("Provider %s failed: %s", provider_name, e)
                vote = ModelVote(
                    provider=provider_name,
                    model_id=provider_name,
                    score=0,
                    total=0,
                    raw_text=f"ERROR: {e}",
                    confidence=0.0,
                )
                return provider_name, vote, None

        with concurrent.futures.ThreadPoolExecutor(max_workers=len(providers)) as pool:
            futures = {
                pool.submit(_grade_with_provider, name): name
                for name in providers
            }
            for future in concurrent.futures.as_completed(futures):
                _name, vote, parsed = future.result()
                model_votes.append(vote)
                if parsed is not None:
                    provider_results.append(parsed)

        # Merge results
        if not provider_results:
            # All providers failed — return empty result
            return GradingResult(
                student_file="",
                total_score=0,
                total_possible=0,
                comments="ERROR: All grading providers failed.",
                model_votes=model_votes,
            )

        result = self._merge_ensemble_results(provider_results, model_votes, mode)
        return result

    def _merge_ensemble_results(
        self,
        results: list[GradingResult],
        votes: list[ModelVote],
        mode: str,
    ) -> GradingResult:
        """
        Merge multiple provider results into a single consensus result.

        Strategy:
        - If scores agree (within divergence threshold): use the most detailed result
        - If scores diverge with an outlier: DO NOT auto-assign — flag for mandatory human review
        - Per-question: majority vote on each question's score
        """
        if len(results) == 1:
            results[0].model_votes = votes
            results[0].consensus_confidence = self._estimate_single_model_confidence(results[0])
            return results[0]

        import statistics

        # Check score agreement
        scores = [r.total_score for r in results]
        totals = [r.total_possible for r in results]
        max_total = max(totals) if totals else 1

        score_range = max(scores) - min(scores)
        divergence = score_range / max_total if max_total > 0 else 0

        # Pick the primary result (most detailed — most question_results)
        primary = max(results, key=lambda r: len(r.question_results) + len(r.section_results))

        if divergence <= self.config.max_score_divergence:
            # Agreement — use primary result with median score
            median_val = round(statistics.median(scores), 1)
            # Clamp to valid bounds
            if median_val < 0:
                median_val = 0
            if max_total > 0 and median_val > max_total:
                median_val = max_total
            primary.total_score = median_val
            primary.consensus_confidence = min(0.95, 0.7 + (1 - divergence) * 0.25)
        else:
            # Divergence detected — check for outliers before trusting median
            median_score = statistics.median(scores)
            has_extreme_outlier = False

            if len(scores) >= 3:
                # Check if any score is an extreme outlier (>40% of total away from median)
                for s in scores:
                    if abs(s - median_score) > (max_total * 0.40):
                        has_extreme_outlier = True
                        break

            if has_extreme_outlier:
                # An extreme outlier could mean one model correctly caught something
                # the others missed, or one model badly misread the submission.
                # Either way, a human MUST decide — auto-grading is not safe here.
                primary.total_score = round(median_score, 1)
                primary.consensus_confidence = 0.0  # Force needs_human_review
                primary.comments += (
                    f"\n\n[ENSEMBLE CRITICAL: Extreme score divergence detected "
                    f"({', '.join(f'{s:.1f}' for s in scores)}). "
                    f"One or more models produced a score far from the consensus. "
                    f"This could indicate a misread or a legitimate grading discrepancy. "
                    f"Median ({median_score:.1f}) assigned provisionally — "
                    f"MANDATORY HUMAN REVIEW REQUIRED before releasing this grade.]"
                )
            else:
                # Moderate divergence — use median but flag for review
                primary.total_score = round(median_score, 1)
                primary.consensus_confidence = max(0.3, 0.7 - divergence)
                primary.comments += (
                    f"\n\n[ENSEMBLE NOTE: Model scores diverged "
                    f"({', '.join(f'{s:.1f}' for s in scores)}). "
                    f"Using median. Human review recommended.]"
                )

        # Per-question cross-validation (exam mode)
        if mode == "exam" and all(r.question_results for r in results):
            primary = self._cross_validate_questions(primary, results)

        primary.model_votes = votes

        # Aggregate token usage
        primary.total_input_tokens = sum(
            v.tokens_used.get("input_tokens", 0) for v in votes
        )
        primary.total_output_tokens = sum(
            v.tokens_used.get("output_tokens", 0) for v in votes
        )
        primary.total_api_calls = len([v for v in votes if v.confidence > 0])

        return primary

    def _cross_validate_questions(
        self,
        primary: GradingResult,
        all_results: list[GradingResult],
    ) -> GradingResult:
        """
        Cross-validate per-question scores across providers.
        Flag questions where providers disagree.
        """
        # Build question lookup for each result
        result_maps = []
        for r in all_results:
            qmap = {q.question_number: q for q in r.question_results}
            result_maps.append(qmap)

        for q in primary.question_results:
            qnum = q.question_number
            scores_for_q = []
            for qmap in result_maps:
                if qnum in qmap:
                    scores_for_q.append(qmap[qnum].points_earned)

            if len(scores_for_q) >= 2:
                if max(scores_for_q) != min(scores_for_q):
                    import statistics
                    median_q = round(statistics.median(scores_for_q), 1)

                    # Check if disagreement is extreme (e.g., 0 vs full credit)
                    q_range = max(scores_for_q) - min(scores_for_q)
                    if q.points_possible > 0 and q_range >= q.points_possible * 0.5:
                        # Major disagreement — one model gave 0 and another gave full
                        # This often means one model misread the handwriting.
                        # Flag for human review instead of trusting median.
                        q.points_earned = median_q
                        q.confidence = ConfidenceLevel.LOW
                        q.reasoning += (
                            f" [CRITICAL: Models strongly disagreed on Q{qnum}: "
                            f"{scores_for_q}. Median used provisionally — "
                            f"human must verify this answer.]"
                        )
                    else:
                        # Minor disagreement — median is reasonable
                        q.points_earned = median_q
                        q.confidence = ConfidenceLevel.LOW
                        q.reasoning += (
                            f" [Models disagreed: {scores_for_q}. Using median.]"
                        )

        # Recalculate total
        primary.total_score = sum(q.points_earned for q in primary.question_results)
        primary.total_score += sum(fr.points_earned for fr in primary.free_response_results)

        return primary

    def _apply_fr_consistency(
        self,
        result: GradingResult,
        page_images: list[dict],
        grading_prompt: str,
        max_long_side: int,
        jpeg_quality: int,
    ) -> GradingResult:
        """
        Re-grade FR questions multiple times and flag disagreements.

        Research shows self-consistency sampling (3-5 passes) catches 5-10%
        more errors on free-response questions. Flags any FR where scores
        differ across passes as LOW confidence for human review.
        """
        passes = self.config.fr_consistency_passes
        logger.info("FR self-consistency: running %d additional passes...", passes)

        fr_scores: dict[int, list[float]] = {
            fr.question_number: [fr.points_earned] for fr in result.free_response_results
        }

        for p in range(passes - 1):  # -1 because we already have the first pass
            try:
                repass = self._grade_single_model(
                    page_images, grading_prompt, max_long_side, jpeg_quality
                )
                for fr in repass.free_response_results:
                    if fr.question_number in fr_scores:
                        fr_scores[fr.question_number].append(fr.points_earned)
            except Exception as e:
                logger.warning("FR consistency pass %d failed: %s", p + 2, e)

        # Flag disagreements
        for fr in result.free_response_results:
            scores = fr_scores.get(fr.question_number, [])
            if len(scores) < 2:
                continue
            score_range = max(scores) - min(scores)
            if score_range > 0 and fr.points_possible > 0:
                # Any score variation on FR = flag for review
                fr.confidence = ConfidenceLevel.LOW
                fr.reasoning += (
                    f" [FR CONSISTENCY: {len(scores)} passes scored "
                    f"{', '.join(str(s) for s in scores)}. "
                    f"Range={score_range:.1f} — human review recommended.]"
                )
                # Use median score as best estimate
                import statistics as _stats
                fr.points_earned = _stats.median(scores)

        # Recalculate total
        result.total_score = sum(q.points_earned for q in result.question_results)
        result.total_score += sum(fr.points_earned for fr in result.free_response_results)

        return result

    def _parse_response(self, raw_text: str, mode: str = "exam") -> GradingResult:
        """
        Parse a grading response — try structured JSON first.

        If JSON parsing fails, attempt legacy text parsing BUT mark the result
        as degraded and requiring mandatory human review. A student's grade
        should never silently degrade to regex-extracted values without review.
        """
        if self.config.structured_output:
            result = self._try_parse_json(raw_text, mode)
            if result:
                return result
            # JSON parsing failed — this is a data integrity concern.
            # The LLM returned something we can't fully parse into structured data.
            logger.warning("Structured JSON parsing failed — result will require human review")

        if self.config.fallback_to_legacy:
            result = parse_legacy_grade_text(raw_text, mode)
            # Mark this result as degraded — the legacy parser loses per-question
            # detail (correct_terms, missing_terms, reasoning, bloom_level, etc.)
            # so we MUST flag it for human verification.
            result.consensus_confidence = 0.0  # Force needs_human_review = True
            result.comments = (
                "[PARSE WARNING: LLM output could not be parsed as structured JSON. "
                "Score was extracted via text pattern matching and may be incomplete or incorrect. "
                "MANDATORY HUMAN REVIEW REQUIRED — verify score and question breakdown before releasing.]\n\n"
                + result.comments
            )
            result.scan_format_notes = (
                (result.scan_format_notes + "\n" if result.scan_format_notes else "")
                + "[RAW LLM OUTPUT PRESERVED FOR REVIEW — see model_votes.raw_text]"
            )
            return result

        # No fallback at all — return explicit failure
        return GradingResult(
            student_file="",
            total_score=0,
            total_possible=0,
            consensus_confidence=0.0,
            comments=(
                "[GRADING FAILED: LLM output could not be parsed. "
                "MANDATORY HUMAN REVIEW REQUIRED. Raw output preserved in model_votes.]\n\n"
                + raw_text[:500]
            ),
        )

    def _try_parse_json(self, raw_text: str, mode: str) -> GradingResult | None:
        """Attempt to parse structured JSON response."""
        # Strip any markdown fencing
        text = raw_text.strip()
        if text.startswith("```"):
            text = re.sub(r"^```\w*\n?", "", text)
            text = re.sub(r"\n?```$", "", text)
        text = text.strip()

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            # Try to find JSON object in the text
            json_match = re.search(r"\{[\s\S]*\}", text)
            if json_match:
                try:
                    data = json.loads(json_match.group())
                except json.JSONDecodeError:
                    return None
            else:
                return None

        if not isinstance(data, dict):
            return None

        try:
            if mode == "exam":
                return self._parse_exam_json(data)
            else:
                return self._parse_notebook_json(data)
        except (KeyError, TypeError, ValueError):
            return None

    def _parse_exam_json(self, data: dict) -> GradingResult:
        """Parse structured exam JSON into GradingResult with validation."""
        questions = []
        for q in data.get("questions", []):
            pts_earned = float(q.get("points_earned", 0))
            pts_possible = float(q.get("points_possible", 0))

            # Clamp impossible values — LLM sometimes hallucinates scores
            raw_earned = pts_earned
            clamp_note = ""
            if pts_earned < 0:
                pts_earned = 0
                clamp_note = f"[Score adjusted from {raw_earned} to 0 — negative value clamped] "
            if pts_possible > 0 and pts_earned > pts_possible:
                pts_earned = pts_possible
                clamp_note = f"[Score adjusted from {raw_earned} to {pts_possible} — exceeds possible] "

            reasoning = q.get("reasoning", "")
            if clamp_note:
                reasoning = clamp_note + reasoning

            questions.append(QuestionResult(
                question_number=q.get("question_number", 0),
                points_earned=pts_earned,
                points_possible=pts_possible,
                confidence=ConfidenceLevel(q.get("confidence", "HIGH").upper()),
                # Vision layer
                raw_reading=q.get("raw_reading", q.get("student_answer", "")),
                alternative_readings=q.get("alternative_readings", []),
                reading_confidence=q.get("reading_confidence", "HIGH"),
                # Scoring layer
                student_answer=q.get("student_answer", ""),
                correct_terms=q.get("correct_terms", []),
                missing_terms=q.get("missing_terms", []),
                wrong_terms=q.get("wrong_terms", []),
                reasoning=reasoning,
                illegible=q.get("illegible", False),
                faint_content=q.get("faint_content", False),
                answer_hunted=q.get("answer_hunted", False),
            ))

        free_response = []
        for fr in data.get("free_response", []):
            fr_earned = float(fr.get("points_earned", 0))
            fr_possible = float(fr.get("points_possible", 0))
            if fr_earned < 0:
                fr_earned = 0
            if fr_possible > 0 and fr_earned > fr_possible:
                fr_earned = fr_possible

            free_response.append(FreeResponseResult(
                question_number=fr.get("question_number", 0),
                points_earned=fr_earned,
                points_possible=fr_possible,
                confidence=ConfidenceLevel(fr.get("confidence", "HIGH").upper()),
                rubric_level_matched=fr.get("rubric_level_matched", ""),
                reasoning=fr.get("reasoning", ""),
            ))

        total_score = float(data.get("total_score", 0))
        total_possible = float(data.get("total_possible", 0))

        # Validate total against sum of parts — catch LLM arithmetic errors
        computed_total = sum(q.points_earned for q in questions) + sum(fr.points_earned for fr in free_response)
        computed_possible = sum(q.points_possible for q in questions) + sum(fr.points_possible for fr in free_response)
        comments = data.get("comments", "")

        if questions or free_response:
            # If we have per-question data, trust the sum over the LLM's stated total
            if abs(computed_total - total_score) > 0.01:
                comments = (
                    f"[SCORE CORRECTION: LLM reported {total_score}/{total_possible} but "
                    f"per-question sum is {computed_total}/{computed_possible}. "
                    f"Discrepancy of {abs(computed_total - total_score):.2f} pts. "
                    f"Using per-question sum — arithmetic verified.]\n\n" + comments
                )
                total_score = computed_total
            if computed_possible > 0:
                total_possible = computed_possible

        # Final cap
        if total_possible > 0 and total_score > total_possible:
            total_score = total_possible

        return GradingResult(
            student_file="",
            grading_mode="exam",
            total_score=total_score,
            total_possible=total_possible,
            question_results=questions,
            free_response_results=free_response,
            shift_detected=data.get("shift_detected", False),
            shift_details=data.get("shift_details", "") or "",
            comments=comments,
            scan_format_notes=data.get("scan_format_notes", "") or "",
        )

    def _parse_notebook_json(self, data: dict) -> GradingResult:
        """Parse structured notebook JSON into GradingResult with validation."""
        sections = []
        for s in data.get("sections", []):
            deductions = []
            for d in s.get("deductions", []):
                deductions.append(Deduction(
                    points=float(d.get("points", 0)),
                    reason=d.get("reason", ""),
                    criterion=d.get("criterion", ""),
                    propagated=d.get("propagated", False),
                ))

            sec_earned = float(s.get("points_earned", 0))
            sec_possible = float(s.get("points_possible", 0))
            if sec_earned < 0:
                sec_earned = 0
            if sec_possible > 0 and sec_earned > sec_possible:
                sec_earned = sec_possible

            sections.append(SectionResult(
                section_name=s.get("section_name", ""),
                points_earned=sec_earned,
                points_possible=sec_possible,
                confidence=ConfidenceLevel(s.get("confidence", "HIGH").upper()),
                deductions=deductions,
                reasoning=s.get("reasoning", ""),
            ))

        total_score = float(data.get("total_score", 0))
        total_possible = float(data.get("total_possible", 0))
        comments = data.get("comments", "")

        # Validate total against sum of sections
        if sections:
            computed_total = sum(s.points_earned for s in sections)
            computed_possible = sum(s.points_possible for s in sections)
            if abs(computed_total - total_score) > 0.01:
                comments = (
                    f"[SCORE CORRECTION: LLM reported {total_score}/{total_possible} but "
                    f"per-section sum is {computed_total}/{computed_possible}. "
                    f"Discrepancy of {abs(computed_total - total_score):.2f} pts. "
                    f"Using per-section sum — arithmetic verified.]\n\n" + comments
                )
                total_score = computed_total
            if computed_possible > 0:
                total_possible = computed_possible

        if total_possible > 0 and total_score > total_possible:
            total_score = total_possible

        return GradingResult(
            student_file="",
            grading_mode="notebook",
            total_score=total_score,
            total_possible=total_possible,
            section_results=sections,
            comments=comments,
            scan_format_notes=data.get("scan_format_notes", "") or "",
        )
