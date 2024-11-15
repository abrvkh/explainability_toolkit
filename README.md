# Ana's Explainability Toolkit 🔎

A collection of methods to i) interpret deep neural network states (the `decode` folder) and ii) act on these interpretations (the `act` folder). 

All methods rely on a model format similar to a HuggingFace🤗 model with the option to `output_hidden_states = True` to output the intermediate states and `output_attentions=True` to output attention maps.

The `notebooks` folder contains notebooks that show examples of how the methods can be used. 

**Decode** 
* **LogitLens** reference: https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens
* **PatchScope** reference: https://arxiv.org/abs/2401.06102 *Coming soon*

**Act** 
* **Steering vectors** reference: https://arxiv.org/abs/2308.10248

**Notebooks**
* LogitLens applied to Phi3: notebooks/phi3_logitlens.ipynb
* Steering vectors applied to Phi3: notebooks/phi3_steering_vectors.ipynb (see the YouTube video here: https://www.youtube.com/watch?v=cp-YSyc5aW8 and https://www.youtube.com/watch?v=cuUUWYtEJKY)
* Controlling the personalities inside your model: 
    * Emotion-based: notebooks/steer_for_emotions.ipynb based on https://arxiv.org/pdf/2310.01405 and https://github.com/andyzoujm/representation-engineering/
    * Persona-based: notebooks/steer_for_personas.ipynb based on https://arxiv.org/pdf/2406.12094 