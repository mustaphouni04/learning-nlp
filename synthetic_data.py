import json
from pathlib import Path
from typing import Dict, Union
from google import genai
from google.genai import types
from random import randrange
import json
import os
import time
from tqdm import tqdm
from dotenv import load_dotenv
from google.api_core import retry
import google

load_dotenv()

class SynthethicGen:
    def __init__(self, datapath: Union[Path, str]):
        with open(datapath) as f:
            data = json.load(f)
        self.data = data

    @retry.Retry(predicate=retry.if_exception_type(google.api_core.exceptions.InternalServerError))
    def generate_text(self, client_api_key: str):
        example_idx = randrange(100)
        client = genai.Client(api_key=client_api_key)
        static = self.data[example_idx]["Text"]
        example = static.replace("*", "")
        response = client.models.generate_content(
            model= "gemini-2.5-flash-preview-05-20",
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

    @retry.Retry(predicate=retry.if_exception_type(google.api_core.exceptions.InternalServerError))
    def generate_summary(self, client_api_key: str, text_sample: Dict[str, str]):
        client = genai.Client(api_key=client_api_key)

        response = client.models.generate_content(
            model= "gemini-2.5-flash-preview-05-20",
            config=types.GenerateContentConfig(
                system_instruction="""Act like a clinical doctor that summarizes medical texts.
                        Generate a summary of the given medical text giving:
                        - Reason of why the patient came to the doctor 
                        - Effects seen during the stay and the verdict on the patient's condition

                        The summary has to be in Spanish and should contain the following:
                        'Paciente que ingresa por...', 'Durante el ingreso...', 'El diagnóstico se orienta a...'
                        These are examples, you can rephrase them.
                        Do not use bullet points and be brief but concise, make it around 5 lines long.
                        """),
                contents=f'Genera un resumen de este texto medico: {text_sample["Text"]}'
            ) 
        print(response.text)
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

    def append_summary_to_json(self, client_api_key: str,
                                     output_file: str = 'output.json',
                                     filename: str = 'data.json'):
        if os.path.exists(output_file):
            with open(output_file, "r") as f:
                try:
                    data_storage = json.load(f)
                except json.JSONDecodeError:
                    data_storage = []
        else:
            data_storage = []
        out_file = open(output_file, "w")
        try:
            with open(filename, 'r+') as file:
                file_data = json.load(file)

            for sample in tqdm(file_data[len(data_storage):], desc="Generating summaries..."):
                sample["Summary"] = self.generate_summary(client_api_key, sample)
                data_storage.append(sample)
                out_file.seek(0)
                json.dump(data_storage, out_file, indent = 4, ensure_ascii = False)
            out_file.close()
        except FileNotFoundError:
            print("Can't append summary to non-existing JSON file")
        
datapath = "uab_summary_2024_all.json"

gen = SynthethicGen(datapath)
api_key = os.environ.get('GEMINI_THIRD_KEY')

gen.append_summary_to_json(client_api_key=api_key)

"""
for i in tqdm(range(500)):
    if i % 10 == 0:
        time.sleep(61)
    gen.append_text_to_json(client_api_key=api_key)
"""



