from llm_sdk.llm_sdk import Small_LLM_Model
import numpy as np
import json

def softmax(x):
    """
    Compute softmax values for each set of scores in x.
    
    Args:
        x: Input array of shape (batch_size, num_classes) or (num_classes,)
        
    Returns:
        Softmax probabilities of same shape as input
    """
    # For numerical stability, subtract the maximum value from each input vector
    # This prevents overflow when calculating exp(x)
    shifted_x = x - np.max(x, axis=-1, keepdims=True)
    
    # Calculate exp(x) for each element
    exp_x = np.exp(shifted_x)
    
    # Calculate the sum of exp(x) for normalization
    sum_exp_x = np.sum(exp_x, axis=-1, keepdims=True)
    
    # Normalize to get probabilities
    probabilities = exp_x / sum_exp_x
    
    return probabilities

def main():
    input = "output a json format of the result that 2 + 5 = , in a json format like {'result' = }"
    llm = Small_LLM_Model()

    def add_next_word(prompt: str) ->str:
    # pour encoder un text (prend un str || return un Torch.tensor)
        input_IDs = llm.encode(prompt).tolist()[0]

    # call le llm pour predire le prochain mot. (return un raw logits).
        logits = llm.get_logits_from_input_ids(input_IDs)

    #  convertir le raw logits en proba avec softmax.
        proba = softmax(logits)

        next_token_id = int(np.argmax(proba))
        
        news_id = input_IDs + [next_token_id]
        return llm.decode(news_id)
    
    word = add_next_word(input)
    word = add_next_word(word)
    word = add_next_word(word)
    word = add_next_word(word)
    word = add_next_word(word)
    word = add_next_word(word)
    word = add_next_word(word)
    print(word)


if __name__ == "__main__":
    main()
