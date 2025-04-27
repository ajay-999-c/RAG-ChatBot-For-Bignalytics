import logging

logging.basicConfig(
    filename="rag_pipeline.log",
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)

def log_event(message: str):
    logging.info(message)

def log_pipeline_step(step_name, input_text, input_tokens, output_tokens, time_taken, section_type=None, retrieval_size=None, user_id=None):
    log_message = f"USER: {user_id} | STEP: {step_name} | INPUT TOKENS: {input_tokens} | OUTPUT TOKENS: {output_tokens} | TIME: {time_taken:.2f}s | SECTION: {section_type} | RETRIEVED DOCS: {retrieval_size} | INPUT: {input_text}"
    logging.info(log_message)
