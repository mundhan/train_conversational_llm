import json
from openai import OpenAI
from pydantic import BaseModel

gpt_output_path = "" ##

class conversation_data(BaseModel):
    conversations: list[dict]
    emotion: str
    location: str
    situation: str
    feature1: str
    feature2: str
    
client = OpenAI(api_key = "") ##

key_values = {
    "location": [''], ##
    "situation": [''], ##
    "feature1": [''], ##
    "feature2": [''], ##
}

def gpt_generate(location, situation, feature1, feature2):
    response = client.chat.completions.create(
        model="gpt-5",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f""}
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "conversation_data",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "conversations": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "role": {
                                        "type": "string",
                                        "enum": ["역할 1", "역할 2"]
                                        },
                                    "content": {"type": "string"}
                                },
                                "required": ["role", "content"],
                                "additionalProperties": False
                            },
                        },
                        "emotion": {"type": "string"},
                        "location": {"type": "string"},
                        "situation": {"type": "string"},
                        "feature1": {"type": "string"},
                        "feature2": {"type": "string"},
                    },
                    "required": ["conversations", "emotion", "location", "situation", "feature1", "feature2"],
                    "additionalProperties": False
                }
            }
        }
    )
    
    return response.choices[0].message.content.strip()

with open(gpt_output_path, "a", encoding="utf-8") as f:
    for loc in key_values["location"]: ##
        for sit in key_values["situation"]: ##
            for feat1 in key_values["feature1"]: ##
                for feat2 in key_values["feature2"]: ##
                    try:
                        gpt_response = gpt_generate(loc, sit, feat1, feat2, is_crim)
                        f.write(gpt_response + "\n")
                    except Exception as e:
                        print(f"Error generating conversation for {loc}, {sit}, {feat1}, {feat2}, {is_crim}: {e}")
    
    
    
    

    
        

