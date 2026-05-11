import os
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate
from langchain_qwq import ChatQwen
from pydantic.v1 import BaseModel
from typing import List

load_dotenv()

os.environ["LANGCHAIN_PROJECT"] = "LangchainDemo"
os.environ["LANGCHAIN_TRACING_V2"] = os.environ.get("LANGCHAIN_TRACING_V2", "false")

# 创建模型
model = ChatQwen(
    model="qwen-turbo",
    api_key=os.environ.get("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    temperature=0.8
)


# 1、定义数据模型
class MedicalBilling(BaseModel):
    patient_id: int
    patient_name: str
    diagnosis_code: str
    procedure_code: str
    total_charge: float
    insurance_claim_amount: float


class MedicalBillingList(BaseModel):
    records: List[MedicalBilling]


# 2、提供样例数据
examples = [
    {"patient_id": 123456, "patient_name": "张娜", "diagnosis_code": "J20.9", "procedure_code": "99203", "total_charge": 500.0, "insurance_claim_amount": 350.0},
    {"patient_id": 789012, "patient_name": "王兴鹏", "diagnosis_code": "M54.5", "procedure_code": "99213", "total_charge": 150.0, "insurance_claim_amount": 120.0},
    {"patient_id": 345678, "patient_name": "刘晓辉", "diagnosis_code": "E11.9", "procedure_code": "99214", "total_charge": 300.0, "insurance_claim_amount": 250.0},
]

# 3、创建提示模板
example_prompt = PromptTemplate(
    input_variables=["patient_id", "patient_name", "diagnosis_code", "procedure_code", "total_charge", "insurance_claim_amount"],
    template="Patient ID: {patient_id}, Patient Name: {patient_name}, Diagnosis Code: {diagnosis_code}, Procedure Code: {procedure_code}, Total Charge: ${total_charge}, Insurance Claim Amount: ${insurance_claim_amount}"
)

prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix="请按照以上格式生成医疗账单数据。",
    suffix="""
请生成 {count} 条新的医疗账单数据，要求：
1. 患者姓名使用中文名字
2. 患者年龄在18-80岁之间
3. 总费用在100-5000元之间
4. 保险索赔金额不超过总费用的80%
5. 诊断代码使用有效的ICD-10编码
6. 不要重复示例数据
""",
    input_variables=["count"],
)

# 4、创建结构化数据生成链
chain = prompt | model.with_structured_output(schema=MedicalBillingList)

# 5、调用生成链
result = chain.invoke({"count": 10})

# 6、输出结果
print("成功生成医疗账单数据：")
for i, record in enumerate(result.records, 1):
    print(f"{i}. Patient ID: {record.patient_id}, Patient Name: {record.patient_name}, "
          f"Diagnosis Code: {record.diagnosis_code}, Procedure Code: {record.procedure_code}, "
          f"Total Charge: ${record.total_charge}, Insurance Claim Amount: ${record.insurance_claim_amount}")
