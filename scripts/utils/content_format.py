from typing import Dict


def generate_fine_tuning(question: str, answer: str) -> Dict[str, str]:
    """
    {"messages": [{"role": "user", "content": question},
                {"role": "assistant", "content": answer}]}"""
    return {
        "messages": [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ]
    }


def generate_pretraining(content: str) -> Dict[str, str]:
    """
    {"text": content }"""
    return {"text": content}


def generate_instruction(question: str, answer: str) -> Dict[str, str]:
    """
    {"instruction": question, "output": answer}"""
    return {"input": question, "response": answer}
