import anthropic
import base64
import cv2
import json
import os
import random
from dotenv import load_dotenv
from openai import OpenAI
from typing import Optional, Tuple
from domain.prompts import CLAUDE_BBOX_PROMPT, OBJECT_DETECTION_PROMPT
from domain.schemas import BoundingBoxes, ModelProvider

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY")



class ObjectDetector:
    def __init__(
        self,
        openai_client: Optional[OpenAI] = None,
        claude_client: Optional[anthropic.Anthropic] = None,
        openai_model: str = "gpt-4.1",
        claude_model: str = "claude-sonnet-4-20250514"
    ):
        """
        Initialize the ObjectDetector with API clients and model configurations.

        Args:
            openai_client: OpenAI client instance
            claude_client: Anthropic client instance
            openai_model: OpenAI model for both vision and text tasks
            claude_model: Claude model for both vision and text tasks
        """
        self.openai_client = openai_client
        self.claude_client = claude_client
        self.openai_model = openai_model
        self.claude_model = claude_model

        if not openai_client and not claude_client:
            raise ValueError("At least one client (OpenAI or Claude) must be provided")

    def _base64_encode_image(self, image_path: str) -> str:
        """Encode image to base64 string."""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def _get_image_media_type(self, image_path: str) -> str:
        """Determine image media type from file extension."""
        if image_path.lower().endswith(('.png',)):
            return "image/png"
        elif image_path.lower().endswith(('.jpg', '.jpeg')):
            return "image/jpeg"
        elif image_path.lower().endswith(('.gif',)):
            return "image/gif"
        elif image_path.lower().endswith(('.webp',)):
            return "image/webp"
        else:
            return "image/jpeg"

    def _generate_bounding_boxes_openai(
        self,
        image_path: str,
        user_query: str,
    ) -> BoundingBoxes:
        """Generate bounding boxes using OpenAI API."""
        if not self.openai_client:
            raise ValueError("OpenAI client not initialized")

        prompt_with_instructions = OBJECT_DETECTION_PROMPT.format(
            USER_INSTRUCTIONS=user_query)

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{self._base64_encode_image(image_path)}",
                            "detail": "high",
                        },
                    },
                    {"type": "text", "text": prompt_with_instructions},
                ],
            },
        ]

        completion = self.openai_client.chat.completions.create(
            model=self.openai_model, messages=messages)

        if not completion.choices[0].message.content:
            raise ValueError("Expected message content on response was not found")

        semi_structured_response = completion.choices[0].message.content

        completion = self.openai_client.beta.chat.completions.parse(
            model=self.openai_model,
            messages=[
                {
                    "role": "user",
                    "content": f"What are the coordinates of all bounding boxes: {semi_structured_response}",
                }
            ],
            response_format=BoundingBoxes,
        )

        bounding_boxes = completion.choices[0].message.parsed
        if not bounding_boxes:
            raise ValueError("No bounding boxes extracted")

        return bounding_boxes
    
    def _generate_bounding_boxes_claude(
        self,
        image_path: str,
        user_query: str,
    ) -> BoundingBoxes:
        """Generate bounding boxes using Claude API."""
        if not self.claude_client:
            raise ValueError("Claude client not initialized")

        prompt_with_instructions = OBJECT_DETECTION_PROMPT.format(
            USER_INSTRUCTIONS=user_query)

        image_data = self._base64_encode_image(image_path)
        media_type = self._get_image_media_type(image_path)

        enhanced_prompt = f"""{prompt_with_instructions}{CLAUDE_BBOX_PROMPT}"""
        message = self.claude_client.messages.create(
            model=self.claude_model,
            max_tokens=4000,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": image_data,
                            },
                        },
                        {
                            "type": "text",
                            "text": enhanced_prompt,
                        },
                    ],
                }
            ],
        )

        response_text = message.content[0].text

        # Extract JSON from response
        # how does anthropic still not have structured responses in claude
        try:
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1

            if start_idx == -1 or end_idx == 0:
                raise ValueError("No JSON found in response")

            json_str = response_text[start_idx:end_idx]
            parsed_data = json.loads(json_str)

            bounding_boxes = BoundingBoxes(**parsed_data)
        except (json.JSONDecodeError, ValueError) as e:
            raise ValueError(f"Failed to parse bounding boxes from Claude response: {e}")

        if not bounding_boxes.boxes:
            raise ValueError("No bounding boxes extracted")        
        return bounding_boxes
    
    def generate_bounding_boxes(
        self,
        image_path: str,
        user_query: str,
        provider: ModelProvider = ModelProvider.OPENAI,
    ) -> BoundingBoxes:
        """
        Generate bounding boxes using the specified provider.
        
        Args:
            image_path: Path to the image file
            user_query: User's query about objects to detect
            provider: Which API provider to use (OpenAI or Claude)
            
        Returns:
            BoundingBoxes object containing detected objects and their coordinates
        """
        if provider == ModelProvider.OPENAI:
            return self._generate_bounding_boxes_openai(
                image_path=image_path, user_query=user_query)
        elif provider == ModelProvider.CLAUDE:
            return self._generate_bounding_boxes_claude(
                image_path=image_path, user_query=user_query)
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    def draw_bounding_boxes(
        self,
        image_path: str,
        bounding_boxes: BoundingBoxes,
        box_thickness: int = 8,
        label_size: float = 1.0,
        output_path: Optional[str] = None,
    ) -> str:
        """
        Draw bounding boxes on the image and display/save it.
        
        Args:
            image_path: Path to the image file
            bounding_boxes: BoundingBoxes object with detection results
            box_thickness: Thickness of the bounding box lines
            label_size: Size of the text labels
            output_path: Path to save the output image. If None, creates one based on input path
                
        Returns:
            Path to the output image file
        """
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Unable to read image file: {image_path}")

        height, width, _ = image.shape

        for box in bounding_boxes.boxes:
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            start_point = (int(box.x1 * width), int(box.y1 * height))
            end_point = (int(box.x2 * width), int(box.y2 * height))

            image = cv2.rectangle(image, start_point, end_point, color, box_thickness)
            label = box.object_name

            text_thickness = max(1, int(label_size * 1.5))
            label_dimensions, _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, label_size, text_thickness
            )
            label_position = (start_point[0], start_point[1] - 20)

            cv2.rectangle(
                image,
                (label_position[0], label_position[1] - label_dimensions[1] - 6),
                (label_position[0] + label_dimensions[0], label_position[1] + 6),
                color,
                -1,
            )

            cv2.putText(
                image,
                label,
                label_position,
                cv2.FONT_HERSHEY_SIMPLEX,
                label_size,
                (255, 255, 255),
                text_thickness,
                cv2.LINE_AA,
            )

        if output_path is None:
            base_name = os.path.splitext(image_path)[0]
            output_path = f"{base_name}_with_bboxes.jpg"

        cv2.imwrite(output_path, image)
        print(f"Image saved to: {output_path}")
        return output_path

    def detect_objects(
        self,
        image_path: str,
        user_prompt: str,
        provider: ModelProvider = ModelProvider.OPENAI,
        output_path: Optional[str] = None,
    ) -> Tuple[BoundingBoxes, str]:
        """
        Complete object detection pipeline: generate bounding boxes and display results.
        
        Args:
            image_path: Path to the image file
            user_prompt: User's query about objects to detect
            object_detection_prompt: Template prompt for object detection
            provider: Which API provider to use (OpenAI or Claude)
            output_path: Path to save the output image
            
        Returns:
            Tuple of (BoundingBoxes object, output_image_path)
        """
        bounding_boxes = self.generate_bounding_boxes(
            image_path=image_path, user_query=user_prompt, provider=provider)
        output_image_path = self.draw_bounding_boxes(
            image_path=image_path, bounding_boxes=bounding_boxes,
            output_path=output_path)
        return bounding_boxes, output_image_path


if __name__ == "__main__":
    openai_client = OpenAI(api_key=OPENAI_API_KEY)
    claude_client = anthropic.Anthropic(api_key=CLAUDE_API_KEY)

    detector = ObjectDetector(
        openai_client=openai_client,
        claude_client=claude_client)

    try:
        results_openai, output_path = detector.detect_objects(
            image_path="./imgs/input/two_zebras.jpg",
            user_prompt="Find all animals in this image",
            provider=ModelProvider.OPENAI,
            output_path="./imgs/output/openai_bbox.jpg"
        )
        print("OpenAI Results:", results_openai)
        print("Output saved to:", output_path)
    except Exception as e:
        print(f"OpenAI detection failed: {e}")

    try:
        results_claude, output_path = detector.detect_objects(
            image_path="./imgs/input/two_zebras.jpg",
            user_prompt="Find all animals in this image",
            provider=ModelProvider.CLAUDE,
            output_path="./imgs/output/claude_bbox.jpg"
        )
        print("Claude Results:", results_claude)
        print("Output saved to:", output_path)
    except Exception as e:
        print(f"Claude detection failed: {e}")