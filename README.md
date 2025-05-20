# MarkMuse - AI Marketing Suite 🌠

**MarkMuse** is a powerful Streamlit application designed to assist marketing professionals in generating a wide array of creative marketing assets. It leverages the capabilities of large language models (LLMs) and image generation AIs to produce text and visuals tailored to specific campaign needs, audience segments, and brand guidelines. This tool features a conceptual multi-agent system to refine content, offers campaign batch generation, and provides advanced prompt engineering options for power users.

## 🚀 Overview

MarkMuse aims to streamline the content creation process by providing a user-friendly interface to interact with sophisticated AI models. Users can define detailed parameters for their marketing content, including:

* **Content Type:** From blog intros to social media updates and ad copy.
* **Campaign Details:** Define overall campaign themes and generate multiple related assets.
* **Audience Targeting:** Specify primary audience and advanced segments (demographics, psychographics, cultural nuances, gender focus).
* **Tone & Style:** Choose from various tones to match brand voice.
* **Language & Length:** Generate content in multiple languages and desired lengths.
* **Brand Guidelines:** Upload policy documents (currently `.txt`) to guide the AI.
* **Visual Context:** Upload existing campaign images for the AI to "see" and understand.
* **AI Image Generation:** Generate accompanying visuals for text assets using Stability AI.
* **Advanced AI Controls:** Select specific AI models, adjust temperature for creativity, and use negative prompts for image generation.

The application simulates a multi-agent AI system where:
1.  A **Generator Agent** creates initial content.
2.  An **Evaluator Agent** refines it for audience, language, and tone.
3.  A **Guideline Enforcer Agent** checks against uploaded policies and reports its actions.

## ✨ Features

* **Single Asset Generation:** Create individual pieces of marketing content with fine-grained control.
* **Campaign Mode / Batch Generation:** Define a central campaign theme and generate multiple, varied content types (e.g., blog intro, social posts, ad copy) in one go.
* **Conceptual Multi-Agent System:** AI is prompted to act as a team of specialized agents (Generator, Evaluator, Guideline Enforcer) for more robust and compliant content.
    * Includes a "Guideline Enforcer Agent Report" detailing changes made for compliance.
* **Advanced Audience Segmentation:** Define target audience by demographics, psychographics, cultural values, and gender focus.
* **Multi-Language Support:** Generate content in various languages.
* **Policy/Brand Guideline Adherence:** Upload `.txt` files containing brand guidelines or policies for the AI to consider.
* **Contextual Image Understanding:** Upload existing campaign images; the AI (OpenAI Vision) describes them to provide visual context for text generation.
* **AI Image Generation:**
    * Generate accompanying images using Stability AI.
    * Provide custom image prompts and select style presets.
    * Use negative prompts to guide image generation.
* **Advanced Prompt Engineering Options (Sidebar):**
    * Select OpenAI model (e.g., `gpt-4o`, `gpt-4-turbo`, `gpt-3.5-turbo`).
    * Adjust OpenAI temperature for creativity.
    * Select Stability AI engine.
* **Editable Output & Refinement:** Edit generated text directly and use an "AI Editor Agent" to refine it further.
* **Session-Based Generation History:** Review the last 5 generations within the current session, including prompts and outputs.
* **API Key Management:** Input API keys via the sidebar for local use.
* **User-Friendly Interface:** Built with Streamlit for an interactive experience.

## 🛠️ Technologies Used

* **Python:** Core programming language.
* **Streamlit:** For building the interactive web application.
* **OpenAI API:** For advanced text generation (`gpt-4o`, `gpt-4-turbo`, `gpt-3.5-turbo`) and image-to-text (`gpt-4-vision-preview`).
* **Stability AI API:** For text-to-image generation (e.g., `stable-diffusion-xl-1024-v1-0`).
* **Pillow (PIL):** For image manipulation (handling image bytes).

## ⚙️ Setup and Installation

1.  **Clone the Repository (if applicable):**
    ```bash
    git clone https://your-repository-url/MarkMuse.git
    cd MarkMuse
    ```

2.  **Create a Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install Dependencies:**
    Make sure you have Python 3.8+ installed.
    ```bash
    pip install streamlit openai stability-sdk Pillow python-dotenv
    ```

4.  **API Key Configuration:**
    * **For Local Development (Option 1 - using `.env` file):**
        * Create a file named `.env` in the root project directory.
        * Add your API keys to this file:
            ```env
            OPENAI_API_KEY="sk-your_openai_key_here"
            STABILITY_API_KEY="sk-your_stability_key_here" 
            ```
        * Uncomment the `dotenv` loading lines in the Python script if you use this method (currently, the script relies on sidebar input).
    * **For Local Development (Option 2 - Sidebar Input - Current Method):**
        * The application allows you to enter API keys directly in the sidebar. These are stored in Streamlit's session state for the current session.
    * **For Deployment (Critical):**
        * **DO NOT** hardcode API keys or commit them to your repository.
        * Use Streamlit's Secrets Management: Create a `.streamlit/secrets.toml` file (add this to `.gitignore`):
            ```toml
            OPENAI_API_KEY = "sk-your_actual_openai_key"
            STABILITY_API_KEY = "sk-your_actual_stability_key"
            ```
        * Access these in your script using `st.secrets["OPENAI_API_KEY"]` and `st.secrets["STABILITY_API_KEY"]`. You would modify the script to use `st.secrets` if keys are found there, otherwise falling back to sidebar input if desired for flexibility.

## ▶️ How to Run the Application

1.  Ensure your virtual environment is activated (if you created one).
2.  Navigate to the project directory in your terminal.
3.  Run the Streamlit application:
    ```bash
    streamlit run your_script_name.py 
    ```
    (Replace `your_script_name.py` with the actual name of the Python file, e.g., `markmuse_app.py`).
4.  The application will open in your default web browser.
5.  Enter your OpenAI and Stability AI API keys in the sidebar to enable AI functionalities.

## 📖 Usage Guide

1.  **Configure API Keys:** Enter your OpenAI and Stability AI API keys in the sidebar.
2.  **Advanced Prompt Engineering (Sidebar):** Adjust OpenAI model, temperature, Stability AI engine, and negative prompt as needed.
3.  **Select Generation Mode:**
    * **Single Asset Generation:** For creating individual pieces of content.
        * Fill in the "Core Content Definition," "Audience & Language," and "Guidelines & Assets" sections.
        * Click "Generate Single Asset."
    * **Campaign Asset Generation:** For creating multiple assets for a cohesive campaign.
        * Define "Campaign Core Details."
        * Set "Campaign Audience & Language" and "Campaign Guidelines & Assets" that apply to all assets in the batch.
        * In "Assets to Generate for this Campaign," select multiple content types.
        * Define a general desired length and key directives for all assets in the campaign.
        * Click "Generate Full Campaign Batch."
4.  **Review Output:**
    * The generated text will appear. You can edit it directly in the text area.
    * The "Guideline Enforcer Agent Report" will detail any changes made for compliance.
    * If image generation was requested, the AI-generated image will be displayed.
5.  **Refine Content:** Click "Refine Content with AI Editor Agent" to send the current (potentially edited) text back to the AI for further improvements.
6.  **View History:** Check the "View Recent Generation History" expander to see your last 5 generations.

### ⚠️ Placeholder AI Logic

The current Python script includes functions like `generate_openai_text_multi_agent`, `describe_image_openai`, and `generate_stability_image`. These functions are set up to make **real API calls** to OpenAI and Stability AI services using the provided API keys.

**There is no placeholder logic for the core AI functionalities in the provided `markmuse_streamlit_v4_campaign_advanced` script. It is intended to work with live APIs.** If API keys are not provided or are invalid, the AI features will fail and display an error.

## 🔮 Future Enhancements

* **Advanced Document Parsing:** Support for `.pdf` and `.docx` policy documents using libraries like `PyPDF2`, `python-docx`, or `Unstructured`.
* **More Granular Campaign Controls:** Allow individual length, tone, and constraint settings for each asset type within a campaign batch.
* **Persistent History & User Accounts:** Save generation history beyond the current session, potentially linked to user accounts (would require a database and authentication).
* **Direct Model Fine-tuning Interface (Ambitious):** Allow users to provide larger datasets to fine-tune models for their specific brand voice.
* **Vector Database for Guidelines:** For very large policy documents, implement a RAG (Retrieval Augmented Generation) system using a vector database for more efficient context retrieval.
* **Integration with Marketing Platforms:** Direct export or posting to social media, email marketing tools, etc.
* **A/B Testing Suggestions:** AI suggests variations of content for A/B testing.
* **Performance Analytics (Conceptual):** If integrated with analytics, AI could learn from past content performance.

## 🤝 Contributing

Contributions are welcome! If you have ideas for improvements or find bugs, please feel free to:
1.  Fork the repository.
2.  Create a new branch (`git checkout -b feature/YourAmazingFeature`).
3.  Make your changes.
4.  Commit your changes (`git commit -m 'Add some YourAmazingFeature'`).
5.  Push to the branch (`git push origin feature/YourAmazingFeature`).
6.  Open a Pull Request.

## 📄 License

This project can be licensed under the MIT License . See the `LICENSE` file for details.

---

Happy Marketing Content Generation with MarkMuse!