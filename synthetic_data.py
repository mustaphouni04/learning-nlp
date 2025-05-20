import json
from pathlib import Path
from typing import Dict, Union
from google import genai
from google.genai import types
from random import randrange
import json
import os
from dotenv import load_dotenv

load_dotenv()

class SynthethicGen:
    def __init__(self, datapath: Union[Path, str]):
        with open(datapath) as f:
            data = json.load(f)
        self.data = data

    def generate_text(self, client_api_key: str):
        example_idx = randrange(100)
        client = genai.Client(api_key=client_api_key)
        static = self.data[example_idx]["Text"]
        example = static.replace("*", "")
        response = client.models.generate_content(
            model="gemini-2.5-flash-preview-05-20",
            config=types.GenerateContentConfig(
                system_instruction="""Act like a clinical doctor writing real medical reports.
                          Write a medical text THE SAME SIZE as the text received about a patient with [condition X].
                          You can think of a different condition.
                          The text should be detailed, clinical, and should include symptoms, diagnosis, and treatment plan.
                          The generated text should be faithful to the received example, if the example has lowercase words, this
                          should be maintained. The generated text should be in Catalan."""),
            contents=f'Donat aquest exemple de text medic: {example}, Genera una versió sintetica de la mateixa mida que el text original:'
        ) 
        return response.text
    
    def append_text_to_json(self, client_api_key: str, filename: str = 'data.json'):
        record = {"Text": self.generate_text(client_api_key)}
        try:
            with open(filename, 'r+') as file:
                try:
                    file_data = json.load(file)
                    if not isinstance(file_data, list):
                        file_data = []
                except json.JSONDecodeError:
                    file_data = []

                file_data.append(record)
                file.seek(0)
                file.truncate()
                json.dump(file_data, file, indent = 4, ensure_ascii=False)
        except FileNotFoundError:
            with open(filename, 'w') as file:
                json.dump([record], file, indent=4, ensure_ascii=False)


datapath = "uab_summary_2024_all.json"

gen = SynthethicGen(datapath)
api_key = os.environ.get('GEMINI_API_KEY')
gen.append_text_to_json(client_api_key=api_key)



