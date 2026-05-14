import torch
from datasets import load_dataset


LABEL_LIST = ["A", "B", "C", "D"]

CONTENT_TEMPLATE = (
    "Question: {question}\n"
    "A) {a}\n"
    "B) {b}\n"
    "C) {c}\n"
    "D) {d}\n"

    "Respond with ONLY the letter of the correct answer, nothing else.\n"
    "Answer:"
)



def make_quality_prompt(ex):

    A, B, C, D = ex["options"]

    return f"""Read the passage and answer the multiple-choice question.

    Passage:
    {ex["article"]}

    Question:
    {ex["question"]}

    Options:
    A. {A}
    B. {B}
    C. {C}
    D. {D}

    Answer with only one letter: A, B, C, or D.

    Answer:"""


def prepare_mmlu(tokenizer, max_examples=2000):
   
    ds = load_dataset("cais/mmlu", "all", split="test")
    examples = []
    
    for ex in ds:
        content = CONTENT_TEMPLATE.format(
            question=ex['question'],
            a=ex['choices'][0],
            b=ex['choices'][1],
            c=ex['choices'][2],
            d=ex['choices'][3],
        )
        content_ids = tokenizer.encode(content, add_special_tokens=False)
        label_letter = LABEL_LIST[ex['answer']]
        

        examples.append({
            'input_ids': torch.tensor(content_ids, dtype=torch.long),
            'label_letter': label_letter
        })

        if len(examples) >= max_examples:
            break

    print(f"[MMLU] Prepared {len(examples)}")

    return examples
