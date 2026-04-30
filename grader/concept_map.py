"""
Multilingual concept equivalence map for biology grading.
Maps English biology terms to their equivalents in Spanish, Mandarin, Korean, Arabic.
"""

MULTILINGUAL_CONCEPTS: dict[str, dict[str, list[str]]] = {
    # Cell biology
    "mitosis": {"es": ["mitosis"], "zh": ["有丝分裂", "有絲分裂"], "ko": ["유사분열", "체세포분열"], "ar": ["الانقسام الفتيلي", "انقسام خيطي"]},
    "meiosis": {"es": ["meiosis"], "zh": ["减数分裂", "減數分裂"], "ko": ["감수분열"], "ar": ["الانقسام الاختزالي"]},
    "cell division": {"es": ["división celular"], "zh": ["细胞分裂"], "ko": ["세포 분열"], "ar": ["انقسام الخلية"]},
    "cell membrane": {"es": ["membrana celular", "membrana plasmática"], "zh": ["细胞膜", "质膜"], "ko": ["세포막"], "ar": ["غشاء الخلية"]},
    "nucleus": {"es": ["núcleo"], "zh": ["细胞核"], "ko": ["핵"], "ar": ["نواة الخلية"]},
    "chromosome": {"es": ["cromosoma"], "zh": ["染色体"], "ko": ["염색체"], "ar": ["كروموسوم"]},

    # PCR / Molecular biology
    "PCR": {"es": ["PCR", "reacción en cadena de la polimerasa"], "zh": ["聚合酶链反应", "PCR"], "ko": ["PCR", "중합효소연쇄반응"], "ar": ["PCR", "تفاعل البوليميراز المتسلسل"]},
    "denaturation": {"es": ["desnaturalización"], "zh": ["变性", "解链"], "ko": ["변성"], "ar": ["تمسخ البروتين"]},
    "annealing": {"es": ["hibridación", "templado"], "zh": ["退火", "引物结合"], "ko": ["어닐링"], "ar": ["الترابط"]},
    "DNA": {"es": ["ADN", "DNA"], "zh": ["DNA", "脱氧核糖核酸"], "ko": ["DNA"], "ar": ["الحمض النووي", "DNA"]},
    "RNA": {"es": ["ARN", "RNA"], "zh": ["RNA", "核糖核酸"], "ko": ["RNA"], "ar": ["الحمض النووي الريبوزي", "RNA"]},

    # Genetics
    "gene": {"es": ["gen"], "zh": ["基因"], "ko": ["유전자"], "ar": ["جين"]},
    "allele": {"es": ["alelo"], "zh": ["等位基因", "等位基因"], "ko": ["대립유전자"], "ar": ["أليل"]},
    "dominant": {"es": ["dominante"], "zh": ["显性"], "ko": ["우성"], "ar": ["سائد"]},
    "recessive": {"es": ["recesivo"], "zh": ["隐性"], "ko": ["열성"], "ar": ["متنحٍ"]},
    "phenotype": {"es": ["fenotipo"], "zh": ["表现型", "表型"], "ko": ["표현형"], "ar": ["النمط الظاهري"]},
    "genotype": {"es": ["genotipo"], "zh": ["基因型"], "ko": ["유전자형"], "ar": ["النمط الجيني"]},

    # Ecology
    "photosynthesis": {"es": ["fotosíntesis"], "zh": ["光合作用"], "ko": ["광합성"], "ar": ["التمثيل الضوئي"]},
    "respiration": {"es": ["respiración"], "zh": ["呼吸作用"], "ko": ["호흡"], "ar": ["التنفس الخلوي"]},
    "ecosystem": {"es": ["ecosistema"], "zh": ["生态系统"], "ko": ["생태계"], "ar": ["النظام البيئي"]},
    "evolution": {"es": ["evolución"], "zh": ["进化", "演化"], "ko": ["진화"], "ar": ["التطور"]},
}

LANGUAGE_GRADING_INSTRUCTIONS: dict[str, str] = {
    "es": (
        "IMPORTANT: This student has answered in Spanish. "
        "Grade for scientific accuracy regardless of language. "
        "Spanish biology terms are valid: 'mitosis'='mitosis', 'ADN'='DNA', "
        "'membrana plasmática'='cell membrane'. "
        "Do not penalize for using Spanish terminology."
    ),
    "zh": (
        "IMPORTANT: This student has answered in Chinese (Mandarin). "
        "Grade for scientific accuracy regardless of language. "
        "Chinese biology terms are valid: '有丝分裂'='mitosis', 'DNA'='DNA', "
        "'细胞膜'='cell membrane'. Do not penalize for using Chinese terminology."
    ),
    "ko": (
        "IMPORTANT: This student has answered in Korean. "
        "Grade for scientific accuracy regardless of language. "
        "Korean biology terms are valid: '유사분열'='mitosis', 'DNA'='DNA', "
        "'세포막'='cell membrane'. Do not penalize for using Korean terminology."
    ),
    "ar": (
        "IMPORTANT: This student has answered in Arabic. "
        "Grade for scientific accuracy regardless of language. "
        "Arabic biology terms are valid: 'الانقسام الفتيلي'='mitosis', "
        "'الحمض النووي'='DNA'. Do not penalize for using Arabic terminology."
    ),
}
