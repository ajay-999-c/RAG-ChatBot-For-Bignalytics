import logging
import os
from datetime import datetime
import csv


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




def save_full_pipeline_log(log_data: dict, user_id: str):
    os.makedirs("full_logs", exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename_txt = f"full_logs/user_{user_id}_{timestamp}.log"
    filename_csv = f"full_logs/user_{user_id}_{timestamp}.csv"

    # Save text log
    with open(filename_txt, "w", encoding="utf-8") as f:
        f.write(f"Question: {log_data['user_query']}\n\n")
        f.write("1️⃣ Query Rewriting ✅\n\n")
        f.write(f"{log_data['rewritten_query']}\n\n")
        f.write("2️⃣ Sub-Questions ✅\n\n")
        for idx, q in enumerate(log_data['sub_questions'], 1):
            f.write(f"{idx}. {q}\n")
        f.write("\n")
        f.write("3️⃣ Retrieval ✅\n\n")
        f.write(f"Chunks Retrieved: {len(log_data['retrieved_chunks'])}\n")
        for idx, chunk in enumerate(log_data['retrieved_chunks'], 1):
            f.write(f"Chunk {idx}: {chunk[:500]}...\n\n")
        f.write("4️⃣ Prompt Context ✅\n\n")
        f.write(f"{log_data['final_prompt'][:1000]}...\n\n")
        f.write("5️⃣ Final Answer ✅\n\n")
        f.write(f"{log_data['generated_answer']}\n\n")
        f.write("6️⃣ Performance Metrics ✅\n\n")
        f.write(f"Input Tokens: {log_data['input_tokens']}\n")
        f.write(f"Output Tokens: {log_data['output_tokens']}\n")
        f.write(f"Total Time Taken: {log_data['total_time']:.2f} seconds\n")

    print(f"✅ Full pipeline log saved at {filename_txt}")

    # Save CSV log
    with open(filename_csv, "w", encoding="utf-8", newline="") as csvfile:
        fieldnames = [
            "Timestamp", "User ID", "Question", 
            "Rewritten Query", "Sub-Questions",
            "Chunks Retrieved", "Prompt Context", 
            "Generated Answer", "Input Tokens", "Output Tokens", "Total Time (s)"
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        writer.writerow({
            "Timestamp": timestamp,
            "User ID": user_id,
            "Question": log_data["user_query"],
            "Rewritten Query": log_data["rewritten_query"],
            "Sub-Questions": " | ".join(log_data["sub_questions"]),
            "Chunks Retrieved": len(log_data["retrieved_chunks"]),
            "Prompt Context": log_data["final_prompt"][:500],  # First 500 chars
            "Generated Answer": log_data["generated_answer"],
            "Input Tokens": log_data["input_tokens"],
            "Output Tokens": log_data["output_tokens"],
            "Total Time (s)": f"{log_data['total_time']:.2f}"
        })

    print(f"✅ CSV version of pipeline log saved at {filename_csv}")
