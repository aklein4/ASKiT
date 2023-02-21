

import torch
from responder import Responder

def main():
    
    model = Responder()
    
    answer = model.forward(("What kind of animals are cats?", "Cats belong to the genus feline. Felines are a type of animal called mammals"))
    
    print(answer)
    
if __name__ == "__main__":
    main()