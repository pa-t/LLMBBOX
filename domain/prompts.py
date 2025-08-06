OBJECT_DETECTION_PROMPT = """Analyze this image and identify all objects based on the following instructions:
{USER_INSTRUCTIONS}

For each object, provide:
1. The object name
2. Normalized bounding box coordinates (x1, y1, x2, y2) where:
    - (0,0) is the top-left corner
    - (1,1) is the bottom-right corner
    - (x1, y1) is the top-left corner of the bounding box
    - (x2, y2) is the bottom-right corner of the bounding box
"""

CLAUDE_BBOX_PROMPT = """


Please provide your response as a JSON object with the following structure:
{{
    "boxes": [
        {{
            "object_name": "name of the object",
            "y1": normalized_y1_coordinate,
            "x1": normalized_x1_coordinate,
            "y2": normalized_y2_coordinate,
            "x2": normalized_x2_coordinate
        }}
    ]
}}

All coordinates should be normalized between 0 and 1, where (0,0) is top-left and (1,1) is bottom-right."""