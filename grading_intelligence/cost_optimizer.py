"""
Cost Optimizer — Intelligent model routing for cost-effective grading.

Routes questions to the cheapest model that can handle them accurately.
Simple MC questions go to Gemini Flash ($0.0005/image), complex reasoning
questions go to Claude ($0.005/image).

Estimated savings: 60-70% on typical mixed exams.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field

from grading_intelligence.structured_output import BloomLevel


# ─── Cost Models ─────────────────────────────────────────────────────────────

class ModelTier(str, Enum):
    """Model tiers ordered by cost."""
    BUDGET = "budget"      # Gemini Flash — ~$0.0005/image
    STANDARD = "standard"  # GPT-4o mini / Gemini Pro — ~$0.002/image
    PREMIUM = "premium"    # Claude Sonnet / GPT-4o — ~$0.005/image


# Approximate cost per 1K input tokens (USD)
MODEL_COSTS = {
    "gemini-2.5-flash": {"input": 0.00015, "output": 0.0006, "tier": ModelTier.BUDGET},
    "gemini-2.5-pro": {"input": 0.00125, "output": 0.005, "tier": ModelTier.STANDARD},
    "gpt-4.1": {"input": 0.002, "output": 0.008, "tier": ModelTier.PREMIUM},
    "gpt-4.1-mini": {"input": 0.0004, "output": 0.0016, "tier": ModelTier.BUDGET},
    "claude-sonnet-4-20250514": {"input": 0.003, "output": 0.015, "tier": ModelTier.PREMIUM},
    "claude-haiku-4-5-20251001": {"input": 0.0008, "output": 0.004, "tier": ModelTier.BUDGET},
}


class QuestionComplexity(str, Enum):
    """Question complexity classification for routing."""
    SIMPLE = "simple"            # MC, true/false, single-word fill-in
    MODERATE = "moderate"        # Short answer, multi-term, matching
    COMPLEX = "complex"          # Free response, calculations, reasoning
    HANDWRITING_CRITICAL = "handwriting_critical"  # Single-letter, ambiguous chars


class RoutingDecision(BaseModel):
    """Model routing decision for a question or question group."""
    question_numbers: list[int]
    complexity: QuestionComplexity
    recommended_tier: ModelTier
    recommended_model: str
    reasoning: str
    estimated_cost_usd: float = 0.0


class CostReport(BaseModel):
    """Cost report for a grading batch."""
    total_submissions: int
    total_questions: int
    routing_decisions: list[RoutingDecision] = Field(default_factory=list)
    estimated_total_cost: float = 0.0
    estimated_savings_vs_premium: float = 0.0  # Percentage saved
    actual_cost: Optional[float] = None
    model_usage: dict[str, int] = Field(default_factory=dict)  # model -> question count


# ─── Question Complexity Classifier ──────────────────────────────────────────

def classify_question_complexity(
    question_text: str,
    answer_text: str = "",
    bloom_level: Optional[BloomLevel] = None,
    num_expected_terms: int = 1,
) -> QuestionComplexity:
    """
    Classify a question's complexity for model routing.

    Rules:
    - Single-letter answers → HANDWRITING_CRITICAL (needs premium for accuracy)
    - MC / true-false / single word → SIMPLE (budget model sufficient)
    - Multi-term, short answer → MODERATE (standard model)
    - Free response, calculations, reasoning → COMPLEX (premium model)
    """
    q_lower = question_text.lower()
    a_lower = answer_text.lower().strip()

    # Single-letter answers need premium (highest error rate)
    if len(a_lower) == 1 and a_lower.isalpha():
        return QuestionComplexity.HANDWRITING_CRITICAL

    # MC indicators
    mc_patterns = ["a)", "b)", "c)", "d)", "a.", "b.", "c.", "d.",
                   "true or false", "t/f", "circle the", "select"]
    if any(p in q_lower for p in mc_patterns):
        return QuestionComplexity.SIMPLE

    # Bloom's level routing
    if bloom_level:
        if bloom_level in (BloomLevel.REMEMBER,):
            return QuestionComplexity.SIMPLE
        elif bloom_level in (BloomLevel.UNDERSTAND, BloomLevel.APPLY):
            return QuestionComplexity.MODERATE
        else:
            return QuestionComplexity.COMPLEX

    # Multi-term answers
    if num_expected_terms >= 3:
        return QuestionComplexity.MODERATE

    # Calculation indicators
    calc_patterns = ["calculate", "compute", "solve", "determine the",
                     "find the value", "show your work"]
    if any(p in q_lower for p in calc_patterns):
        return QuestionComplexity.COMPLEX

    # Free response indicators
    fr_patterns = ["explain", "describe", "discuss", "analyze",
                   "compare and contrast", "evaluate", "justify"]
    if any(p in q_lower for p in fr_patterns):
        return QuestionComplexity.COMPLEX

    # Default
    if num_expected_terms == 1:
        return QuestionComplexity.SIMPLE

    return QuestionComplexity.MODERATE


def route_to_model(complexity: QuestionComplexity) -> tuple[ModelTier, str]:
    """
    Route a question to the appropriate model tier.

    Returns (tier, recommended_model_id).
    """
    routing = {
        QuestionComplexity.SIMPLE: (ModelTier.BUDGET, "gemini-2.5-flash"),
        QuestionComplexity.MODERATE: (ModelTier.STANDARD, "gemini-2.5-pro"),
        QuestionComplexity.COMPLEX: (ModelTier.PREMIUM, "claude-sonnet-4-20250514"),
        QuestionComplexity.HANDWRITING_CRITICAL: (ModelTier.PREMIUM, "claude-sonnet-4-20250514"),
    }
    return routing.get(complexity, (ModelTier.PREMIUM, "claude-sonnet-4-20250514"))


def plan_batch_routing(
    questions: list[dict],  # [{"number": 1, "text": "...", "answer": "...", "points": 3}]
    budget_mode: bool = False,
) -> CostReport:
    """
    Plan optimal model routing for a batch of questions.

    Groups questions by complexity tier to minimize API calls
    (one call per tier rather than per question).
    """
    decisions = []
    tier_groups: dict[ModelTier, list[int]] = {}

    for q in questions:
        complexity = classify_question_complexity(
            q.get("text", ""),
            q.get("answer", ""),
            num_expected_terms=q.get("points", 1),
        )

        if budget_mode:
            # Force everything to budget tier
            tier, model = ModelTier.BUDGET, "gemini-2.5-flash"
        else:
            tier, model = route_to_model(complexity)

        qnum = q.get("number", 0)
        if tier not in tier_groups:
            tier_groups[tier] = []
        tier_groups[tier].append(qnum)

        decisions.append(RoutingDecision(
            question_numbers=[qnum],
            complexity=complexity,
            recommended_tier=tier,
            recommended_model=model,
            reasoning=f"Q{qnum}: {complexity.value} complexity",
        ))

    # Estimate costs
    # Rough estimate: ~2000 tokens per image page, ~1000 tokens output per question
    estimated_total = 0.0
    model_usage = {}
    for d in decisions:
        costs = MODEL_COSTS.get(d.recommended_model, MODEL_COSTS["claude-sonnet-4-20250514"])
        q_cost = (costs["input"] * 2 + costs["output"] * 1) / 1000  # Per question
        d.estimated_cost_usd = round(q_cost, 5)
        estimated_total += q_cost
        model_usage[d.recommended_model] = model_usage.get(d.recommended_model, 0) + 1

    # Compare to all-premium
    premium_costs = MODEL_COSTS["claude-sonnet-4-20250514"]
    all_premium = len(questions) * (premium_costs["input"] * 2 + premium_costs["output"] * 1) / 1000
    savings = ((all_premium - estimated_total) / all_premium * 100) if all_premium > 0 else 0

    return CostReport(
        total_submissions=1,
        total_questions=len(questions),
        routing_decisions=decisions,
        estimated_total_cost=round(estimated_total, 4),
        estimated_savings_vs_premium=round(savings, 1),
        model_usage=model_usage,
    )
