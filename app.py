import streamlit as st
from text_generator import TextGenerator
from transformers import GPT2LMHeadModel, GPT2Tokenizer

st.set_page_config(page_title="Text Generation App", page_icon="📝")

@st.cache_resource
def load_model():
    return TextGenerator()
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    return tokenizer, model

@st.cache_resource
def load_generator():
    return TextGenerator()

generator = load_generator()
       

generator = load_model()

# def generate_text(prompt, max_length=100):
#     inputs = tokenizer.encode(prompt, return_tensors="pt")
#     outputs = model.generate(inputs, max_length=max_length, num_return_sequences=1, temperature=0.7)
#     generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
#     return generated_text

st.title("📝 Text Generation App")

prompt_type = st.selectbox("Choose prompt type:", ["Story", "Article", "Poem"])

if prompt_type == "Story":
    st.subheader("Generate a Creative Story")
    col1, col2 = st.columns(2)
    with col1:
        protagonist = st.text_input("Main character's name:")
        setting = st.text_input("Story setting (time and place):")
    with col2:
        occupation = st.text_input("Character's occupation:")
        genre = st.selectbox("Genre:", ["Fantasy", "Sci-Fi", "Mystery", "Romance", "Adventure", "Horror"])
    
    conflict = st.text_area("Central conflict or challenge:")
    tone = st.select_slider("Story tone:", options=["Humorous", "Dramatic", "Suspenseful", "Whimsical", "Dark"])
    
    prompt = f"""Create a compelling {genre} short story with these elements:
    1. Protagonist: {protagonist}, a {occupation}
    2. Setting: {setting}
    3. Central Conflict: {conflict}
    4. Tone: {tone}

    Begin the story in media res. Develop the character through actions and dialogue. Include vivid sensory details. Build tension around the central conflict and resolve it unexpectedly. Maintain a {tone.lower()} tone throughout. Aim for a satisfying conclusion that leaves a lasting impression."""

elif prompt_type == "Article":
    st.subheader("Generate an Article")
    topic = st.text_input("Article topic:")
    style = st.selectbox("Writing style:", ["Informative", "Persuasive", "Entertaining"])
    prompt = f"Write a {style.lower()} article about {topic}. Include relevant facts, engaging examples, and a clear structure with an introduction, body, and conclusion."

else:
    st.subheader("Generate a Poem")
    theme = st.text_input("Poem theme:")
    form = st.selectbox("Poetic form:", ["Sonnet", "Haiku", "Free Verse"])
    prompt = f"Compose a {form.lower()} poem about {theme}. Use vivid imagery, evocative language, and appropriate structure for the chosen form."

max_length = st.slider("Maximum length of generated text:", 50, 500, 200)

if st.button("Generate Text"):
    with st.spinner("Generating..."):
        generated_text = generator.generate_text(prompt, max_length)
    st.subheader("Generated Text:")
    st.write(generated_text)

st.markdown("---")
st.markdown("Created by Kaustubh Holla")

def main():
    generator = TextGenerator()
    prompt = "Write a unique and diverse article about sports, mentioning various platforms only once:"
    
    generated_text = generator.generate_text(prompt)
    print(generated_text)

# if st.button("Generate Text", key="generate_button"):
#     generated_text = generator.generate_text(prompt, max_length)
#     st.write("Generated Text:")
#     st.write(generated_text)   

if __name__ == "__main__":
    main()

  