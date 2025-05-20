import streamlit as st
import time
import openai
from stability_sdk import client as stability_client
import stability_sdk.interfaces.gooseai.generation.generation_pb2 as generation
from PIL import Image
import io
import base64

# --- Page Configuration ---
st.set_page_config(
    page_title="MarkMuse - AI Marketing Suite",
    page_icon="🌠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Initialize Session State ---
# (Ensure all relevant session state variables are initialized here)
default_session_state = {
    'generation_history': [],
    'current_text_output': "",
    'current_agent_report': "",
    'current_image_prompt': "",
    'current_image_url': None,
    'current_image_caption': "",
    'openai_api_key': "",
    'stability_api_key': "",
    'openai_model': "gpt-4o", # Default OpenAI Model
    'openai_temperature': 0.7, # Default Temperature
    'stability_engine': "stable-diffusion-xl-1024-v1-0", # Default Stability Engine
    'stability_negative_prompt': ""
}
for key, value in default_session_state.items():
    if key not in st.session_state:
        st.session_state[key] = value


# --- API Helper Functions ---
def get_openai_client(api_key):
    if not api_key: return None
    try: return openai.OpenAI(api_key=api_key)
    except Exception as e: st.error(f"Failed to initialize OpenAI client: {e}"); return None

def generate_openai_text_multi_agent(client, prompt_details, model="gpt-4o", temperature=0.7):
    if not client: return "OpenAI client not initialized.", "Agent report not generated: OpenAI client missing."
    system_message = """
You are an AI Marketing Assistant composed of three specialized agents working in sequence:
1.  **Generator Agent:** Creates the initial marketing copy based on core requirements.
2.  **Evaluator Agent:** Refines the draft for language, cultural nuances, target audience alignment, and tone.
3.  **Guideline Enforcer Agent:** Ensures the refined draft strictly adheres to provided business policies, brand guidelines, and explicit constraints. This agent MUST explicitly list any changes made to ensure compliance or state 'No changes needed for guideline adherence.' if applicable.

Produce the final marketing content followed by a SEPARATE report from the Guideline Enforcer Agent.
The report should be clearly marked and enclosed like this:
<GuidelineEnforcerReport>
[Report Content Here]
</GuidelineEnforcerReport>
"""
    user_prompt = "Generate marketing content following these detailed instructions, processed by your internal agents:\n\n"
    user_prompt += f"## Core Request (for Generator Agent for content type: {prompt_details['content_type'].replace('_', ' ').title()}):\n"
    if prompt_details.get('campaign_name'):
        user_prompt += f"- Overall Campaign Name/Theme: {prompt_details['campaign_name']}\n"
        user_prompt += f"- Core Campaign Message: {prompt_details.get('core_campaign_message', 'N/A')}\n"
    user_prompt += f"- Specific Topic for this Asset: {prompt_details['topic']}\n"
    user_prompt += f"- Campaign Approach: {prompt_details['campaign_approach']}\n"
    user_prompt += f"- Desired Length for this Asset: {prompt_details['actual_length_instruction']}\n"
    user_prompt += f"- Output Language: {prompt_details['language']} (Full name: {prompt_details['language_full_name']})\n"

    user_prompt += "\n## Audience & Style Guidelines (for Evaluator Agent):\n"
    user_prompt += f"- Desired Tone for this Asset: {prompt_details['tone']}\n"
    user_prompt += f"- Primary Target Audience: {prompt_details['target_audience']}\n"
    if prompt_details.get('demographics'): user_prompt += f"- Target Demographics: {prompt_details['demographics']}\n"
    if prompt_details.get('culture'): user_prompt += f"- Target Psychographics/Culture: {prompt_details['culture']}\n"
    if prompt_details.get('gender_focus') != 'any': user_prompt += f"- Gender Focus: {prompt_details['gender_focus']}\n"

    user_prompt += "\n## Constraints & Policies (for Guideline Enforcer Agent):\n"
    if prompt_details.get('constraints'): user_prompt += f"- Key Message/CTA/Specific Inclusions/Exclusions for this Asset: {prompt_details['constraints']}\n"
    if prompt_details.get('policy_content'):
        user_prompt += f"\n--- Adhere to the following Business Policy/Brand Guidelines ---\n{prompt_details['policy_content']}\n---------------------------------------------------------------\n"
    if prompt_details.get('uploaded_image_descriptions'):
        user_prompt += f"\n--- Context from Uploaded Campaign Images ---\n" + "\n".join(prompt_details['uploaded_image_descriptions']) + "\n---------------------------------------------\n"
    
    if prompt_details.get('custom_text_to_refine'):
        user_prompt = f"An AI Editor Agent is now active. Please review and refine the following marketing content. The original goal was for a '{prompt_details.get('content_type', 'marketing piece')}' about '{prompt_details.get('topic', 'the given subject')}' as part of campaign '{prompt_details.get('campaign_name', 'N/A')}'. Adhere to any provided policy content. Content to refine:\n\n---\n{prompt_details['custom_text_to_refine']}\n---\n\nProvide the refined content, followed by the Guideline Enforcer Agent Report detailing your changes or confirming adherence."

    user_prompt += "\nBegin generation. Remember to provide the Guideline Enforcer Agent Report separately after the main content, enclosed in <GuidelineEnforcerReport> tags."

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": system_message}, {"role": "user", "content": user_prompt}],
            temperature=temperature
        )
        full_response_content = response.choices[0].message.content.strip()
        report_start_tag = "<GuidelineEnforcerReport>"
        report_end_tag = "</GuidelineEnforcerReport>"
        main_content_final = full_response_content
        agent_report_final = "Guideline Enforcer Agent did not provide a report in the expected format."
        report_start_index = full_response_content.find(report_start_tag)
        report_end_index = full_response_content.find(report_end_tag)

        if report_start_index != -1 and report_end_index != -1 and report_start_index < report_end_index:
            main_content_final = full_response_content[:report_start_index].strip()
            agent_report_final = full_response_content[report_start_index + len(report_start_tag):report_end_index].strip()
        elif report_start_index != -1:
            main_content_final = full_response_content[:report_start_index].strip()
            agent_report_final = full_response_content[report_start_index + len(report_start_tag):].strip() + "\n(Warning: End report tag missing)"
        main_content_final = main_content_final.replace(report_start_tag, "").replace(report_end_tag, "").strip()
        return main_content_final, agent_report_final
    except openai.APIError as e: return f"Error generating text: {e}", f"Agent report failed: {e}"
    except Exception as e: return f"Unexpected error: {e}", f"Agent report failed: {e}"

def describe_image_openai(client, image_bytes, model="gpt-4-vision-preview"):
    if not client: return "OpenAI client not initialized for image description."
    try:
        base64_image = base64.b64encode(image_bytes).decode('utf-8')
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": [
                {"type": "text", "text": "Describe this image in detail for marketing context. Focus on elements, style, mood, and potential message relevant for creating related marketing copy."},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
            ]}], max_tokens=300
        )
        return response.choices[0].message.content.strip()
    except Exception as e: return f"Error describing image: {e}"

def generate_stability_image(api_key, prompt, engine_id="stable-diffusion-xl-1024-v1-0", negative_prompt="", width=1024, height=1024, style_preset=None):
    if not api_key: return None, "Stability AI API Key not provided."
    stability_api = stability_client.StabilityInference(key=api_key, verbose=False, engine=engine_id)
    try:
        answers = stability_api.generate(
            prompt=prompt,
            negative_prompt=negative_prompt if negative_prompt else None,
            steps=30, cfg_scale=7.0, width=width, height=height, samples=1,
            style_preset=style_preset if style_preset != "none" else None
        )
        for resp in answers:
            for artifact in resp.artifacts:
                if artifact.finish_reason == generation.FILTER: return None, "Image generation failed due to safety filters."
                if artifact.type == generation.ARTIFACT_IMAGE:
                    img = Image.open(io.BytesIO(artifact.binary))
                    return img, f"AI-generated visual for: \"{prompt}\""
        return None, "No image generated by Stability AI."
    except Exception as e: return None, f"Error generating image: {e}"

def parse_uploaded_document(uploaded_file):
    if uploaded_file is None: return ""
    try:
        if uploaded_file.type == "text/plain":
            content = uploaded_file.getvalue().decode("utf-8")
            if len(content) > 15000:
                st.warning("Uploaded document is very long. Truncating to first 15,000 characters.")
                return content[:15000]
            return content
        else: st.warning(f"Unsupported file type: {uploaded_file.type}. Only .txt processed."); return ""
    except Exception as e: st.error(f"Error parsing document '{uploaded_file.name}': {e}"); return ""

def add_to_history(prompt_details, text_output, agent_report, image_prompt, image_url_bytes, image_caption, advanced_settings):
    history_item = {
        "prompt_details": prompt_details, "text_output": text_output, "agent_report": agent_report,
        "image_prompt": image_prompt, "image_url_bytes": image_url_bytes, "image_caption": image_caption,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), **advanced_settings # Add advanced settings to history
    }
    st.session_state.generation_history.insert(0, history_item)
    if len(st.session_state.generation_history) > 5: st.session_state.generation_history.pop()

# --- Sidebar ---
st.sidebar.header("API Key Configuration 🔑")
st.sidebar.markdown("Use `st.secrets` for deployed apps!")
st.session_state.openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password", value=st.session_state.openai_api_key, key="sidebar_openai_key_input")
st.session_state.stability_api_key = st.sidebar.text_input("Stability AI API Key", type="password", value=st.session_state.stability_api_key, key="sidebar_stability_key_input")

st.sidebar.header("Advanced Prompt Engineering ⚙️")
st.session_state.openai_model = st.sidebar.selectbox(
    "OpenAI Model (Text Generation):",
    ("gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"), # Add more as needed
    index=["gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"].index(st.session_state.openai_model),
    key="sidebar_openai_model_select"
)
st.session_state.openai_temperature = st.sidebar.slider(
    "OpenAI Temperature (Creativity):", 0.0, 2.0, st.session_state.openai_temperature, 0.1,
    key="sidebar_openai_temp_slider"
)
st.session_state.stability_engine = st.sidebar.selectbox(
    "Stability AI Engine (Image Generation):",
    ("stable-diffusion-xl-1024-v1-0", "stable-diffusion-v1-6", "stable-diffusion-xl-beta-v2-2-2"), # Example engines
    index=["stable-diffusion-xl-1024-v1-0", "stable-diffusion-v1-6", "stable-diffusion-xl-beta-v2-2-2"].index(st.session_state.stability_engine),
    key="sidebar_stability_engine_select"
)
st.session_state.stability_negative_prompt = st.sidebar.text_area(
    "Stability AI Negative Prompt:",
    value=st.session_state.stability_negative_prompt,
    placeholder="e.g., blurry, watermark, text, ugly, deformed",
    height=100,
    key="sidebar_stability_neg_prompt_area"
)

# --- Main Application Interface ---
st.title("🌠 MarkMuse - AI Marketing Suite")
st.markdown("Leverage specialized AI agents and advanced controls to craft precise, compliant, and impactful marketing assets.")
st.markdown("---")

generation_mode = st.radio("Select Generation Mode:", ("Single Asset Generation", "Campaign Asset Generation"), key="gen_mode_radio", horizontal=True)
st.markdown("---")

# Shared form elements
language_options_map = {'en': 'English (US)', 'en_gb': 'English (UK)', 'es': 'Español (Spanish)', 'fr': 'Français (French)', 'de': 'Deutsch (German)', 'ja': '日本語 (Japanese)', 'zh_cn': '中文 (Simplified Chinese)', 'hi': 'हिन्दी (Hindi)', 'pt_br': 'Português (Brazilian)', 'ar': 'العربية (Arabic)', 'ko': '한국어 (Korean)', 'it': 'Italiano (Italian)'}

if generation_mode == "Single Asset Generation":
    st.header("Single Asset Generation Mode")
    with st.form("single_asset_form"):
        st.subheader("1. Core Content Definition")
        topic_single = st.text_input("Product/Service/Specific Topic:", placeholder="e.g., 'New feature X for InnovateX SaaS'", key="topic_single_input")
        campaign_approach_single = st.selectbox("Campaign Approach Context:", ('Thematic', 'Programmatic', 'Hybrid', 'N/A'), key="campaign_approach_single_select")
        uploaded_campaign_images_single = st.file_uploader("Upload Existing Image(s) for Context (Optional):", type=['png', 'jpg', 'jpeg', 'webp'], accept_multiple_files=True, key="campaign_img_uploader_single")
        
        col1_s, col2_s = st.columns(2)
        with col1_s: content_type_single = st.selectbox("Content Type:", ('blog_intro', 'social_media_update', 'product_description', 'website_headline', 'email_campaign_body', 'ad_copy_long', 'video_script_hook'), format_func=lambda x: x.replace('_', ' ').title(), key="content_type_single_select")
        with col2_s: tone_single = st.selectbox("Desired Tone:", ('professional', 'persuasive', 'friendly', 'urgent', 'informative', 'playful', 'empathetic', 'witty', 'luxurious', 'technical'), format_func=lambda x: x.replace('_', ' ').title(), key="tone_single_select")
        
        st.markdown("---")
        st.subheader("2. Audience & Language")
        target_audience_single = st.text_input("Primary Target Audience:", placeholder="e.g., Tech-savvy early adopters", key="target_audience_single_input")
        with st.expander("Advanced Audience Segments..."):
            demographics_single = st.text_input("Key Demographics:", placeholder="e.g., Age: 30-55", key="demographics_single_input")
            culture_single = st.text_input("Psychographics/Values:", placeholder="e.g., Values innovation", key="culture_single_input")
            gender_focus_single = st.selectbox("Gender Focus:", ('any', 'female', 'male', 'neutral'), format_func=lambda x: {'any': 'Any/All', 'female': 'Female-Identifying', 'male': 'Male-Identifying', 'neutral': 'Gender-Neutral'}.get(x), key="gender_focus_single_select")

        col3_s, col4_s = st.columns(2)
        with col3_s:
            length_option_single = st.selectbox("Desired Length:", ('very_short', 'short', 'medium', 'long', 'custom'), index=2, format_func=lambda x: x.replace('_', ' ').title(), key="length_single_select")
            custom_length_input_single = ""
            if length_option_single == 'custom': custom_length_input_single = st.text_input("Specify Custom Length:", key="custom_length_single_input")
        with col4_s: language_code_single = st.selectbox("Output Language:", list(language_options_map.keys()), format_func=lambda x: language_options_map[x], key="language_single_select")
        
        st.markdown("---")
        st.subheader("3. Guidelines & Assets")
        policy_document_file_single = st.file_uploader("Upload Policy/Brand Guidelines (.txt):", type=['txt'], key="policy_doc_uploader_single")
        generate_image_cb_single = st.checkbox("Generate AI Image?", key="gen_img_cb_single")
        image_prompt_input_single = ""
        stability_style_preset_single = "none"
        if generate_image_cb_single:
            image_prompt_input_single = st.text_input("AI Image Prompt:", placeholder="e.g., Diverse team in modern office", key="img_prompt_input_single")
            stability_style_preset_single = st.selectbox("Image Style Preset:", ("none", "photographic", "digital-art", "comic-book", "fantasy-art", "anime", "cinematic"), format_func=lambda x: x.replace('-', ' ').title(), key="style_preset_single_select")
        constraints_input_single = st.text_area("Key Message/CTA/Inclusions/Exclusions:", placeholder="e.g., Emphasize ease-of-use. CTA: 'Learn More'.", height=100, key="constraints_single_area")
        
        submitted_single = st.form_submit_button("✨ Generate Single Asset", use_container_width=True)

    if submitted_single:
        # Validation and processing logic for single asset
        oai_key, stab_key = st.session_state.openai_api_key, st.session_state.stability_api_key
        valid = True
        if not oai_key: st.error("OpenAI API Key missing."); valid = False
        if not topic_single.strip(): st.error("Specific Topic missing."); valid = False
        if generate_image_cb_single and not image_prompt_input_single.strip(): st.error("AI Image Prompt missing."); valid = False
        if generate_image_cb_single and not stab_key: st.error("Stability AI Key missing for image."); valid = False

        if valid:
            openai_client = get_openai_client(oai_key)
            if openai_client:
                with st.spinner("AI Agents crafting single asset..."):
                    policy_content = parse_uploaded_document(policy_document_file_single)
                    img_descs = []
                    if uploaded_campaign_images_single:
                        for i, img_file in enumerate(uploaded_campaign_images_single):
                            with st.spinner(f"Analyzing uploaded image {i+1}..."):
                                img_descs.append(describe_image_openai(openai_client, img_file.getvalue()))
                    
                    actual_length = custom_length_input_single if length_option_single == 'custom' and custom_length_input_single else length_option_single
                    prompt_details = {
                        'content_type': content_type_single, 'topic': topic_single, 'campaign_approach': campaign_approach_single,
                        'tone': tone_single, 'target_audience': target_audience_single, 'demographics': demographics_single,
                        'culture': culture_single, 'gender_focus': gender_focus_single, 'actual_length_instruction': actual_length,
                        'language': language_code_single, 'language_full_name': language_options_map[language_code_single],
                        'constraints': constraints_input_single, 'policy_content': policy_content,
                        'uploaded_image_descriptions': img_descs, 'campaign_name': "N/A (Single Asset)"
                    }
                    text, report = generate_openai_text_multi_agent(openai_client, prompt_details, model=st.session_state.openai_model, temperature=st.session_state.openai_temperature)
                    st.session_state.current_text_output = text
                    st.session_state.current_agent_report = report
                    st.session_state.current_image_url, st.session_state.current_image_caption = None, ""
                    st.session_state.current_image_prompt = image_prompt_input_single if generate_image_cb_single else ""

                    if generate_image_cb_single and image_prompt_input_single and stab_key:
                        with st.spinner("AI Image Agent generating visual..."):
                            img_obj, cap = generate_stability_image(stab_key, image_prompt_input_single, engine_id=st.session_state.stability_engine, negative_prompt=st.session_state.stability_negative_prompt, style_preset=stability_style_preset_single)
                            if img_obj:
                                buffered = io.BytesIO(); img_obj.save(buffered, format="PNG")
                                st.session_state.current_image_url = buffered.getvalue()
                                st.session_state.current_image_caption = cap
                    
                    adv_settings_history = {"openai_model": st.session_state.openai_model, "openai_temperature": st.session_state.openai_temperature, "stability_engine": st.session_state.stability_engine, "stability_negative_prompt": st.session_state.stability_negative_prompt}
                    add_to_history(prompt_details, text, report, st.session_state.current_image_prompt, st.session_state.current_image_url, st.session_state.current_image_caption, adv_settings_history)
                    st.balloons(); st.success("Single asset generated!")


elif generation_mode == "Campaign Asset Generation":
    st.header("Campaign Asset Generation Mode")
    with st.form("campaign_form"):
        st.subheader("A. Campaign Core Details")
        campaign_name_campaign = st.text_input("Campaign Name/Theme:", placeholder="e.g., 'InnovateX Spring Launch'", key="campaign_name_input_campaign")
        core_campaign_message_campaign = st.text_area("Core Campaign Message/Overall Theme:", placeholder="e.g., 'Empowering SMBs with next-gen AI tools for growth.'", height=75, key="core_message_campaign_area")
        
        st.subheader("B. Campaign Audience & Language (Applies to all assets in campaign)")
        target_audience_campaign = st.text_input("Primary Target Audience for Campaign:", placeholder="e.g., Marketing Managers in SMBs", key="target_audience_campaign_input")
        with st.expander("Advanced Audience Segments for Campaign..."):
            demographics_campaign = st.text_input("Key Demographics for Campaign:", placeholder="e.g., Age: 28-45, Tech-savvy", key="demographics_campaign_input")
            culture_campaign = st.text_input("Psychographics/Values for Campaign:", placeholder="e.g., Seek efficiency, data-driven", key="culture_campaign_input")
            gender_focus_campaign = st.selectbox("Gender Focus for Campaign:", ('any', 'female', 'male', 'neutral'), format_func=lambda x: {'any': 'Any/All', 'female': 'Female-Identifying', 'male': 'Male-Identifying', 'neutral': 'Gender-Neutral'}.get(x), key="gender_focus_campaign_select")
        language_code_campaign = st.selectbox("Output Language for All Assets:", list(language_options_map.keys()), format_func=lambda x: language_options_map[x], key="language_campaign_select")

        st.subheader("C. Campaign Guidelines & Assets (Applies to all assets in campaign)")
        policy_document_file_campaign = st.file_uploader("Upload Campaign Policy/Brand Guidelines (.txt):", type=['txt'], key="policy_doc_uploader_campaign")
        uploaded_campaign_images_campaign = st.file_uploader("Upload Existing Campaign Image(s) for Overall Context (Optional):", type=['png', 'jpg', 'jpeg', 'webp'], accept_multiple_files=True, key="campaign_img_uploader_campaign")
        
        st.subheader("D. Assets to Generate for this Campaign")
        content_types_campaign = st.multiselect(
            "Select Content Types for this Campaign:",
            ('blog_intro', 'social_media_update', 'product_description', 'website_headline', 'email_campaign_body', 'ad_copy_long', 'video_script_hook'),
            format_func=lambda x: x.replace('_', ' ').title(),
            key="content_types_campaign_multiselect"
        )
        # For simplicity, using global length and constraints for all batch items in this version
        length_campaign = st.selectbox("Desired Length for ALL Campaign Assets:", ('very_short', 'short', 'medium', 'long'), index=2, format_func=lambda x: x.replace('_', ' ').title(), key="length_campaign_select")
        constraints_campaign = st.text_area("Key Directives/Constraints for ALL Campaign Assets:", placeholder="e.g., Maintain an optimistic tone. Include link to landing page.", height=100, key="constraints_campaign_area")
        
        generate_images_for_campaign_cb = st.checkbox("Generate a unique AI Image for EACH text asset in the campaign?", key="gen_img_campaign_cb")
        # Note: Individual image prompts per asset type in a campaign is a more advanced feature not added here for brevity.
        # A generic prompt based on campaign + asset type could be used, or a single image prompt for the whole campaign.
        # For now, if checked, we might use the campaign topic + asset type as a basic image prompt.
        
        submitted_campaign = st.form_submit_button("🚀 Generate Full Campaign Batch", use_container_width=True)

    if submitted_campaign:
        oai_key_camp, stab_key_camp = st.session_state.openai_api_key, st.session_state.stability_api_key
        valid_camp = True
        if not oai_key_camp: st.error("OpenAI API Key missing."); valid_camp = False
        if not campaign_name_campaign.strip(): st.error("Campaign Name/Theme missing."); valid_camp = False
        if not content_types_campaign: st.error("Please select at least one Content Type for the campaign."); valid_camp = False
        if generate_images_for_campaign_cb and not stab_key_camp: st.error("Stability AI Key missing for image generation."); valid_camp = False

        if valid_camp:
            openai_client_camp = get_openai_client(oai_key_camp)
            if openai_client_camp:
                st.session_state.current_text_output = "" # Clear previous single asset output
                st.session_state.current_agent_report = ""
                st.session_state.current_image_url = None
                st.session_state.current_image_caption = ""
                
                campaign_outputs = []
                with st.spinner(f"AI Agents generating {len(content_types_campaign)} assets for campaign '{campaign_name_campaign}'... This may take a while."):
                    policy_content_camp = parse_uploaded_document(policy_document_file_campaign)
                    img_descs_camp = []
                    if uploaded_campaign_images_campaign:
                        for i, img_file_camp in enumerate(uploaded_campaign_images_campaign):
                            with st.spinner(f"Analyzing campaign image {i+1}..."):
                                img_descs_camp.append(describe_image_openai(openai_client_camp, img_file_camp.getvalue()))
                    
                    for asset_idx, current_content_type_camp in enumerate(content_types_campaign):
                        st.info(f"Generating asset {asset_idx+1}/{len(content_types_campaign)}: {current_content_type_camp.replace('_',' ').title()}...")
                        
                        # For campaign mode, the "topic" for each asset is the overall campaign name/theme
                        # Individual asset constraints could be more specific if UI allowed, here using campaign constraints
                        prompt_details_camp = {
                            'content_type': current_content_type_camp, 
                            'topic': campaign_name_campaign, # Using campaign name as the primary topic for each asset
                            'campaign_name': campaign_name_campaign,
                            'core_campaign_message': core_campaign_message_campaign,
                            'campaign_approach': "N/A (Set at campaign level)", # Assuming campaign approach is campaign-wide, not per asset
                            'tone': "N/A (Set at campaign level)", # Assuming tone is campaign-wide
                            'target_audience': target_audience_campaign, 
                            'demographics': demographics_campaign,
                            'culture': culture_campaign, 
                            'gender_focus': gender_focus_campaign, 
                            'actual_length_instruction': length_campaign, # Using campaign-wide length
                            'language': language_code_campaign, 
                            'language_full_name': language_options_map[language_code_campaign],
                            'constraints': constraints_campaign, # Using campaign-wide constraints
                            'policy_content': policy_content_camp,
                            'uploaded_image_descriptions': img_descs_camp
                        }
                        
                        text_camp, report_camp = generate_openai_text_multi_agent(openai_client_camp, prompt_details_camp, model=st.session_state.openai_model, temperature=st.session_state.openai_temperature)
                        
                        img_url_camp, img_cap_camp = None, ""
                        img_prompt_for_asset = ""
                        if generate_images_for_campaign_cb and stab_key_camp:
                            # Simple image prompt: campaign name + asset type + core message
                            img_prompt_for_asset = f"{campaign_name_campaign} - {current_content_type_camp.replace('_',' ').title()}. Visual representing: {core_campaign_message_campaign[:100]}"
                            with st.spinner(f"Generating image for {current_content_type_camp.replace('_',' ').title()}..."):
                                img_obj_camp, cap_camp = generate_stability_image(stab_key_camp, img_prompt_for_asset, engine_id=st.session_state.stability_engine, negative_prompt=st.session_state.stability_negative_prompt, style_preset="photographic") # Defaulting style for campaign
                                if img_obj_camp:
                                    buffered_camp = io.BytesIO(); img_obj_camp.save(buffered_camp, format="PNG")
                                    img_url_camp = buffered_camp.getvalue()
                                    img_cap_camp = cap_camp
                        
                        campaign_outputs.append({
                            "content_type_title": current_content_type_camp.replace('_',' ').title(),
                            "text_output": text_camp, "agent_report": report_camp,
                            "image_prompt": img_prompt_for_asset, "image_url_bytes": img_url_camp, "image_caption": img_cap_camp
                        })
                        adv_settings_history_camp = {"openai_model": st.session_state.openai_model, "openai_temperature": st.session_state.openai_temperature, "stability_engine": st.session_state.stability_engine, "stability_negative_prompt": st.session_state.stability_negative_prompt}
                        add_to_history(prompt_details_camp, text_camp, report_camp, img_prompt_for_asset, img_url_camp, img_cap_camp, adv_settings_history_camp)

                st.session_state.campaign_generated_outputs = campaign_outputs # Store for display
                st.balloons(); st.success(f"Campaign '{campaign_name_campaign}' assets generated!")


# --- Display Logic (Common for Single and Campaign, adapted for Campaign) ---
if generation_mode == "Campaign Asset Generation" and 'campaign_generated_outputs' in st.session_state and st.session_state.campaign_generated_outputs:
    st.header(f"Generated Assets for Campaign: '{st.session_state.generation_history[0]['prompt_details']['campaign_name']}'")
    for i, asset in enumerate(st.session_state.campaign_generated_outputs):
        st.subheader(f"Asset {i+1}: {asset['content_type_title']}")
        st.text_area(f"Content Output##asset_{i}", asset['text_output'], height=200, key=f"campaign_text_{i}", disabled=True) # Not editable for now in batch
        if asset.get('agent_report'):
            with st.expander(f"Guideline Enforcer Agent Report##asset_{i}", expanded=False):
                st.markdown(f"<div style='background-color:#f0f2f6; padding:10px; border-radius:5px;'><pre style='white-space: pre-wrap; word-wrap: break-word;'>{asset['agent_report']}</pre></div>", unsafe_allow_html=True)
        if asset.get('image_url_bytes'):
            st.image(asset['image_url_bytes'], caption=asset.get('image_caption', f"Image for {asset['content_type_title']}"), use_column_width=True)
        st.markdown("---")

elif generation_mode == "Single Asset Generation" and st.session_state.current_text_output:
    st.subheader("Generated Marketing Content (Single Asset):")
    edited_text_display_area = st.text_area("Content Output (Editable):", value=st.session_state.current_text_output, height=300, key="editable_text_output_display_area_single", help="You can edit the generated text here.")
    if edited_text_display_area != st.session_state.current_text_output: 
        st.session_state.current_text_output = edited_text_display_area

    if st.button("✍️ Refine Content with AI Editor Agent...", key="refine_btn_single"):
        oai_key_refine_single = st.session_state.get("openai_api_key")
        if not oai_key_refine_single: st.error("OpenAI API Key needed for refinement.")
        else:
            openai_client_refine_single = get_openai_client(oai_key_refine_single)
            if openai_client_refine_single and st.session_state.generation_history:
                original_prompt_details_single = st.session_state.generation_history[0]['prompt_details']
                refinement_prompt_details_single = {**original_prompt_details_single, 'custom_text_to_refine': st.session_state.current_text_output}
                with st.spinner("AI Editor Agent refining content..."):
                    refined_output_single, refined_agent_report_single = generate_openai_text_multi_agent(openai_client_refine_single, refinement_prompt_details_single, model=st.session_state.openai_model, temperature=st.session_state.openai_temperature)
                    st.session_state.current_text_output = refined_output_single
                    st.session_state.current_agent_report = refined_agent_report_single
                    st.rerun()

    if st.session_state.current_agent_report:
        with st.expander("View Guideline Enforcer Agent Report", expanded=True):
            st.markdown(f"<div style='background-color:#f0f2f6; padding:10px; border-radius:5px;'><b>--- GUIDELINE ENFORCER AGENT REPORT ---</b><br><pre style='white-space: pre-wrap; word-wrap: break-word;'>{st.session_state.current_agent_report}</pre><b>--- END OF REPORT ---</b></div>", unsafe_allow_html=True)

    col_rate1_single, col_rate2_single, _ = st.columns([1,1,5])
    with col_rate1_single:
        if st.button("👍 Helpful", key="helpful_btn_single"): st.toast("Thanks for your feedback!", icon="😊")
    with col_rate2_single:
        if st.button("👎 Not Helpful", key="unhelpful_btn_single"): st.toast("Thanks! We'll try to improve.", icon="😕")

    if st.session_state.current_image_url:
        st.subheader("Generated AI Image (Single Asset):")
        st.image(st.session_state.current_image_url, caption=st.session_state.current_image_caption, use_column_width=True)

    if uploaded_campaign_images_single and submitted_single: # Show uploaded images after single submission
        st.subheader("Your Uploaded Images (for context):")
        cols_uploaded_img_s = st.columns(min(len(uploaded_campaign_images_single), 4))
        for idx, uploaded_file_disp_item_s in enumerate(uploaded_campaign_images_single):
            cols_uploaded_img_s[idx % 4].image(uploaded_file_disp_item_s, caption=uploaded_file_disp_item_s.name, width=150)


# --- Generation History (Common) ---
if st.session_state.generation_history:
    with st.expander("📜 View Recent Generation History (Last 5)", expanded=False):
        for i, item in enumerate(st.session_state.generation_history):
            st.markdown(f"**Generation {i+1} ({item['timestamp']})** - Model: {item.get('openai_model', 'N/A')}, Temp: {item.get('openai_temperature', 'N/A')}")
            st.markdown(f"*Asset Type: {item['prompt_details'].get('content_type','N/A').replace('_',' ').title()} for Campaign/Topic: {item['prompt_details'].get('campaign_name') or item['prompt_details'].get('topic')}*")
            
            st.text_area(f"Text Output##history_{i}", item['text_output'], height=150, disabled=True, key=f"history_text_area_item_{i}")
            if item.get('agent_report'):
                    st.markdown(f"Agent Report##history_{i}")
                    st.text(item['agent_report'])
            if item.get('image_url_bytes'):
                st.image(item['image_url_bytes'], caption=item.get('image_caption', f"Image for gen {i+1}"), width=200, key=f"history_img_area_item_{i}")
            st.markdown("---")

st.markdown("---")
st.markdown(
        "<div style='text-align: center; color: gray;'>"
        "© 2025 MarkMuse. All rights reserved. · Powered by Streamlit"
        "</div>",
        unsafe_allow_html=True
    )

