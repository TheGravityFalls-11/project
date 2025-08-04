import os
from llama_parse import LlamaParse
from dotenv import load_dotenv
load_dotenv()
os.environ["LLAMA_CLOUD_API_KEY"] = os.getenv("LLAMA_CLOUD_API_KEY")

class DataExtractor:
    def __init__(self):
        self.parser = LlamaParse(result_type="markdown")

    def from_pdf(self, file_path):
        return self.parser.load_data(file_path)

    def from_url(self, url):
        return self.parser.load_data(url)
