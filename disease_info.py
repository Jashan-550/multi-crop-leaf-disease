"""
Educational disease descriptions for TomatoGuard-CNN.
Neutral horticultural information only — not professional advice.
"""

DISCLAIMER = (
    "This tool is for educational purposes only and is NOT a substitute for "
    "professional agricultural diagnosis. Always confirm findings with a "
    "qualified agronomist or plant pathologist before taking action."
)

DISEASE_INFO = {
    "Tomato___Bacterial_spot": (
        "Bacterial spot is caused by Xanthomonas species and appears as small, "
        "dark, water-soaked lesions on leaves and fruit. It spreads more easily "
        "in warm, humid conditions and through splashing water."
    ),
    "Tomato___Early_blight": (
        "Early blight is caused by the fungus Alternaria solani and typically "
        "shows as concentric ring spots on older leaves. It is common in warm, "
        "humid weather and often affects lower foliage first."
    ),
    "Tomato___healthy": (
        "Healthy tomato leaves are generally uniform green with no significant "
        "spots, curling, or discoloration. Regular monitoring helps catch "
        "problems early when they do appear."
    ),
    "Tomato___Late_blight": (
        "Late blight is caused by Phytophthora infestans and can spread rapidly "
        "in cool, wet conditions. Affected leaves often show large, irregular "
        "dark patches and may wilt quickly."
    ),
    "Tomato___Leaf_Mold": (
        "Leaf mold is caused by Passalora fulva (formerly Fulvia fulva) and "
        "often appears as yellow patches on the upper leaf surface with fuzzy "
        "growth underneath. It is more common in humid greenhouse environments."
    ),
    "Tomato___Septoria_leaf_spot": (
        "Septoria leaf spot is caused by Septoria lycopersici and produces "
        "small circular spots with dark borders and light centers on leaves. "
        "It usually starts on lower leaves and can spread upward over time."
    ),
    "Tomato___Spider_mites Two-spotted_spider_mite": (
        "Two-spotted spider mites (Tetranychus urticae) are tiny arachnids that "
        "feed on leaf cells, causing stippling, bronzing, and fine webbing. "
        "They thrive in hot, dry conditions."
    ),
    "Tomato___Target_Spot": (
        "Target spot is caused by Corynespora cassiicola and appears as circular "
        "lesions with concentric rings, resembling a bullseye pattern. It can "
        "affect leaves, stems, and fruit under warm, humid conditions."
    ),
    "Tomato___Tomato_mosaic_virus": (
        "Tomato mosaic virus (ToMV) causes mottled light and dark green patterns "
        "on leaves, sometimes with distortion or stunted growth. It is spread "
        "through contact with infected plants, tools, or hands."
    ),
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": (
        "Tomato yellow leaf curl virus (TYLCV) is transmitted by whiteflies and "
        "causes upward curling of leaves with yellowing and reduced plant vigor. "
        "Infected plants may show stunted growth over time."
    ),
}


def prettify(class_key: str) -> str:
    """Convert raw class key to a human-readable display name."""
    if class_key == "Tomato___healthy":
        return "Healthy"

    name = class_key.replace("Tomato___", "")
    name = name.replace("_", " ")

    # Handle the spider mite class name
    if "spider mite" in name.lower():
        return "Spider Mites (Two-Spotted Spider Mite)"

    return name.title()
