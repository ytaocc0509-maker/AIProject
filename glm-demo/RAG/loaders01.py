from langchain_community.document_loaders import CSVLoader

loader = CSVLoader(file_path='../../../../百度网盘/BaiduNetdiskDownload/2/08_国产大模型 ChatGLM 深度实战/00_课程资料/glm-demo/glm-demo/weather_district_id.csv', encoding='utf-8')

data = loader.load()

for record in data[:2]:
    print(record)