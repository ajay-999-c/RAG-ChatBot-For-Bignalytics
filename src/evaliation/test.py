from deepeval.metrics import FaithfulnessMetric
from deepeval.test_case import LLMTestCase
from pipeline.rag import ask

gold_fee = "₹45,000"
answer, ctx = ask("How much does the Masters Program in Data Science & Analytics cost?")

case = LLMTestCase(
    input="fee query", actual_output=answer,
    retrieval_context=[c.page_content for c in ctx],
    expected_output=gold_fee
)

faith = FaithfulnessMetric(threshold=0.8, model="mistralai/Mistral-7B-Instruct-v0.1")
def test_fee():
    faith.evaluate([case])
