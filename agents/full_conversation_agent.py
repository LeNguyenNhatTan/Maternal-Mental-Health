import os
import time
from utils.llm_client_base import LLMClient
from utils.rag_engine import RAGEngine
import json
import re

class FullConversationAgent:
    def __init__(self, provider="ollama", provider_options=None, model="qwen3:4b", 
                 patient_profile=None, questions=None, rag_engine=None, questionnaire_name=None,
                 disable_rag_evaluation=False):
        """
        Initialize the Full Conversation Agent.
        
        Args:
            provider (str): LLM provider to use (ollama or groq)
            provider_options (dict, optional): Options to pass to the provider client
            model (str): Model to use with the provider
            patient_profile (str): Name of the patient profile to use
            questions (list): List of questions from the questionnaire
            rag_engine (RAGEngine, optional): RAG engine for document retrieval
            questionnaire_name (str, optional): Name of the questionnaire being used
            disable_rag_evaluation (bool): Whether to disable RAG evaluation
        """
        # Initialize default provider options
        if provider_options is None:
            provider_options = {}
            
            # Set default options based on provider
            if provider == "ollama":
                provider_options["base_url"] = "http://localhost:11434"
                
        # Create the client using the factory method
        self.client = LLMClient.create(provider, **provider_options)
        self.model = model
        # Store the original profile name
        self.profile_name = patient_profile
        # Load the actual profile content from file
        self.patient_profile = self._load_profile(patient_profile)
        self.questions = questions
        self.rag_engine = rag_engine
        self.questionnaire_name = questionnaire_name
        self.disable_rag_evaluation = disable_rag_evaluation
        
        # Store the provider name for special handling
        self.provider = provider
        
        # Track documents that have already been seen to avoid duplication
        self.seen_documents = set()
        self.conversation_history = []
        # Initialize responses attribute
        self.responses = []
        
    def _load_profile(self, profile_name):
        """Load a patient profile from file."""
        if not profile_name:
            return "No specific patient profile provided."
        
        profiles_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "profiles")
        
        # Try to load from the profiles directory
        profile_path = os.path.join(profiles_dir, f"{profile_name}.txt")
        
        if os.path.exists(profile_path):
            with open(profile_path, 'r') as f:
                profile_content = f.read()
            return profile_content
        else:
            print(f"Warning: Profile '{profile_name}' not found. Using default profile.")
            return f"Profile named {profile_name} (profile details not found)"
    
    
    def generate_conversation(self):
        """Generate a full conversation between the full conversation agent and patient."""
        print("[DEBUG] Retrieving full questionnaire document")
        
        # Get the full questionnaire document if available
        full_questionnaire_content = ""
        if self.questionnaire_name:
            # Try direct file reading approach first if this is a filename with .txt extension
            if self.questionnaire_name.endswith('.txt'):
                try:
                    # Construct path to questionnaire file
                    questionnaire_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "documents", "questionnaires")
                    questionnaire_path = os.path.join(questionnaire_dir, self.questionnaire_name)
                    
                    # Read file directly
                    if os.path.exists(questionnaire_path):
                        print(f"[DEBUG] Reading questionnaire directly from file: {questionnaire_path}")
                        with open(questionnaire_path, 'r') as f:
                            full_questionnaire_content = f.read()
                        print(f"[DEBUG] Successfully loaded questionnaire file ({len(full_questionnaire_content)} chars)")
                except Exception as e:
                    print(f"[DEBUG] Error reading questionnaire file directly: {str(e)}")
            
            # Fall back to RAG if direct file read failed or if not a .txt file
            if not full_questionnaire_content and self.rag_engine:
                print("[DEBUG] Falling back to RAG engine for questionnaire retrieval")
                specific_docs = self.rag_engine.get_context_for_question(f"full text of {self.questionnaire_name}")
                
                # Handle both old format (list) and new format (dictionary)
                if isinstance(specific_docs, dict) and "content" in specific_docs:
                    # New format from enhanced RAG engine
                    full_questionnaire_content = specific_docs["content"]
                elif isinstance(specific_docs, list) and len(specific_docs) > 0:
                    # Old format (list of strings)
                    full_questionnaire_content = specific_docs[0]
                else:
                    # Default empty string if no content found
                    full_questionnaire_content = ""
        
        # Create a prompt for the conversation generation
        base_conversation_prompt = f"""
        You are an agent which generates a full conversation between a postpartum mental health professional and a patient according to a questionnaire.
        
        IMPORTANT INSTRUCTIONS:
        - Use ONLY the actual name of the questionnaire as provided in the user message
        - The questionnaire may be a general mental health assessment or focused on various conditions
        - Your role is to administer the questionnaire without making diagnostic assumptions up front
        
        Profile for the Mental Health Professional:
        You are a professional postpartum mental health assistant tasked with conducting a psychological assessment interview. Your job is to:

            1. Ask questions from a the questionnaire in a compassionate, professional manner
            2. Respond appropriately to the patient's answers with empathy and understanding

            Important guidelines:
            - Maintain professional boundaries while being empathetic
            - Ask one question at a time
            - Do not make assumptions about the patient's condition before completing the full assessment
            - Use clinical judgment when interpreting responses
            - Consider multiple potential diagnoses before making your final assessment
            - Be thorough and methodical in your approach
            - Provide evidence-based recommendations

            When asking, always use the exact wording from the questionnaire, but deliver it in a warm and empathetic way, so the question feels natural and compassionate.
            Keep your tone professional but warm, showing empathy while maintaining clinical objectivity.
            Make sure to correctly identify and name the specific questionnaire you're administering.
        """
        
        # Special handling for Groq provider - more structured prompt
        if hasattr(self, 'provider') and self.provider == "groq":
            conversation_prompt = base_conversation_prompt + """
            
            YOUR RESPONSE MUST BE STRUCTURED AS A VALID JSON ARRAY OF OBJECTS.
            Each object must have 'role' and 'content' fields.
            The 'role' must be either 'assistant' for the mental health professional or 'patient' for the patient.
            The 'content' field must contain the actual message.
            
            You MUST adhere to proper JSON syntax with double quotes around property names and string values.
            Do not include any explanatory text, commentary, or code blocks around the JSON.
            """
        else:
            conversation_prompt = base_conversation_prompt

        # Define a specific user prompt that clearly asks for JSON format
        base_user_prompt = f"""
        Please generate a full conversation between the mental health professional and the patient.
        
        Here is the full questionnaire document you will be administering:
        
        ```
        {full_questionnaire_content}
        ```
        
        Profile for the Patient:
        {self.patient_profile}
        
        EXTREMELY IMPORTANT:
        - The postpartum mental health professional MUST ask ALL the questions from the questionnaire in order
        - Use the EXACT wording of the questions as they appear in the questionnaire
        - Do NOT skip any questions or add additional diagnostic and screening questions
        - Do not stop after the first question
        - Make sure all {len(self.questions)} questions from the questionnaire are covered in the conversation
        - The patient should respond in NATURAL CONVERSATIONAL LANGUAGE, not with numerical ratings
        - Patient responses should be descriptive and elaborate on their experiences, not just "3" or "4"
        - Patient should describe their symptoms in their own words while addressing the severity implied by the questionnaire
        
        YOUR RESPONSE MUST BE A VALID JSON ARRAY of objects, each with 'role' and 'content' fields.
        
        The format MUST be exactly as follows, with proper JSON syntax:
        ```json
        [
            {{
                "role": "assistant",
                "content": "Introduction and first question"
            }},
            {{
                "role": "patient",
                "content": "Patient response in natural language, NOT numerical ratings"
            }},
            ...and so on until all questions are asked and answered
        ]
        ```
        
        DO NOT include any text before or after the JSON array.
        DO NOT include backticks or "json" markers.
        ONLY return the JSON array itself.
        
        The introduction should:
        - Introduces yourself as a mental health professional
        - Identifies the specific questionnaire you're using by name (from the document)
        - Explains the purpose of this specific assessment 
        - Reassures the patient about confidentiality and creating a safe space
        - Briefly explains how the assessment will proceed ({len(self.questions)} questions)
        - Indicates you're ready to begin with the first question
        """
        
        # Special handling for Groq provider
        if hasattr(self, 'provider') and self.provider == "groq":
            user_prompt = f"""
            Generate a full conversation between the mental health professional and patient as a JSON array.
            
            Here is the full questionnaire document you will be administering:
            
            ```
            {full_questionnaire_content}
            ```
            
            Profile for the Patient:
            {self.patient_profile}
            
            EXTREMELY IMPORTANT:
            - The postpartum mental health professional MUST ask ALL the questions from the questionnaire in order
            - Use the EXACT wording of the questions as they appear in the questionnaire
            - Do NOT skip any questions or add additional diagnostic and screening questions
            - Make sure all {len(self.questions)} questions from the questionnaire are covered in the conversation
            - The patient should respond in NATURAL CONVERSATIONAL LANGUAGE, not with numerical ratings
            - Patient responses should be descriptive and elaborate on their experiences, not just "3" or "4"
            - Patient should describe their symptoms in their own words while addressing the severity implied by the questionnaire
            
            The introduction should:
            - Introduces yourself as a mental health professional
            - Identifies the specific questionnaire you're using by name (from the document above)
            - Explains the purpose of this specific assessment 
            - Reassures the patient about confidentiality and creating a safe space
            - Briefly explains how the assessment will proceed ({len(self.questions)} questions)
            - Indicates you're ready to begin with the first question
            
            CRITICAL: Return ONLY a raw JSON array with no text before or after. 
            Each object must have 'role' and 'content' fields. Include all {len(self.questions)} questions.
            
            Format: [{{\"role\":\"assistant\",\"content\":\"...\"}},{{\"role\":\"patient\",\"content\":\"Patient response in natural language, NOT numerical ratings\"}},...]
            """
        else:
            user_prompt = base_user_prompt
        
        # Create a temporary conversation for generating the introduction
        temp_conversation = [
            {"role": "system", "content": conversation_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        # Generate the conversation using the LLM
        result = self.client.chat(self.model, temp_conversation)
        conversation_text = result['response']
        
        # Clear existing conversation history
        self.conversation_history = []
        
        # Enhanced JSON extraction and cleaning
        def extract_and_clean_json(text):
            print(f"DEBUG: Raw conversation text length: {len(text)}")
            # Log the first and last 100 characters to see start/end format
            if len(text) > 200:
                print(f"DEBUG: Text starts with: {text[:100]}")
                print(f"DEBUG: Text ends with: {text[-100:]}")
            else:
                print(f"DEBUG: Full text: {text}")
            
            # Clean the response text to get valid JSON
            cleaned_text = text.strip()
            
            # Remove backticks, "json" markers, and other common prefixes/suffixes
            cleaned_text = re.sub(r'^```json\s*', '', cleaned_text)
            cleaned_text = re.sub(r'^```\s*', '', cleaned_text)
            cleaned_text = re.sub(r'\s*```$', '', cleaned_text)
            
            # First approach: Try to extract array with regex
            # Remove any text before the first '[' and after the last ']'
            array_match = re.search(r'\[.*\]', cleaned_text, re.DOTALL)
            if array_match:
                cleaned_text = array_match.group(0)
                print(f"DEBUG: Found JSON array with regex")
            
            # Clean control characters and Unicode quotes
            cleaned_text = re.sub(r'[\x00-\x1F\x7F]', '', cleaned_text)
            cleaned_text = cleaned_text.replace('\u201c', '"').replace('\u201d', '"')
            cleaned_text = cleaned_text.replace('\u2018', "'").replace('\u2019', "'")
            
            # Fix common JSON syntax errors
            # Replace single quotes with double quotes (only for keys and string values)
            cleaned_text = re.sub(r'([{,])\s*\'([^\']+)\'\s*:', r'\1"\2":', cleaned_text)
            cleaned_text = re.sub(r':\s*\'([^\']+)\'\s*([,}])', r':"\1"\2', cleaned_text)
            
            # If there's still no proper JSON array structure, try harder to extract one
            if not (cleaned_text.startswith('[') and cleaned_text.endswith(']')):
                print("DEBUG: No proper JSON array found, trying harder extraction")
                # Try to find anything that looks like JSON objects and build an array
                objects = re.findall(r'{.*?}', cleaned_text, re.DOTALL)
                if objects:
                    cleaned_text = "[" + ",".join(objects) + "]"
                    print(f"DEBUG: Built JSON array from {len(objects)} extracted objects")
            
            return cleaned_text
            
        # Clean and try to parse JSON
        cleaned_text = extract_and_clean_json(conversation_text)
        
        try:
            # Try to parse the cleaned JSON
            print(f"DEBUG: Attempting to parse JSON of length {len(cleaned_text)}")
            conversation_data = json.loads(cleaned_text)
            print(f"DEBUG: Successfully parsed JSON with {len(conversation_data)} items")
            
            # Validate that it's a list of properly structured messages
            if isinstance(conversation_data, list):
                for i, message in enumerate(conversation_data):
                    if not isinstance(message, dict):
                        print(f"DEBUG: Item {i} is not a dictionary: {message}")
                        continue
                    
                    if "role" not in message or "content" not in message:
                        print(f"DEBUG: Item {i} missing role or content: {message}")
                        continue
                    
                    role = message.get("role", "")
                    content = message.get("content", "")
                    
                    # Validate and standardize roles
                    if any(r in role.lower() for r in ["assistant", "clinician", "therapist", "professional", "mental health"]):
                        role = "assistant"
                    elif any(r in role.lower() for r in ["user", "patient"]):
                        role = "patient"
                    else:
                        print(f"DEBUG: Invalid role '{role}' at item {i}, defaulting to 'assistant'")
                        role = "assistant"
                    
                    # Ensure content is a string
                    if not isinstance(content, str):
                        content = str(content)
                    
                    # Add the message to the conversation history
                    self.conversation_history.append({
                        "role": role,
                        "content": content
                    })
                
                print(f"DEBUG: Added {len(self.conversation_history)} messages to conversation history")
                
                # Extract responses from conversation history after successful parsing
                self._extract_responses_from_conversation_history()
            else:
                print(f"DEBUG: Parsed JSON is not a list: {type(conversation_data)}")
                # Try to handle single message case
                if isinstance(conversation_data, dict) and "role" in conversation_data and "content" in conversation_data:
                    print(f"DEBUG: Found single message dict, adding to conversation")
                    self.conversation_history.append({
                        "role": conversation_data["role"],
                        "content": conversation_data["content"]
                    })
                elif isinstance(conversation_data, dict):
                    # Try to extract role-content pairs from flat dict structure
                    for role_key, content in conversation_data.items():
                        role = role_key.lower()
                        if "assistant" in role or "clinician" in role or "therapist" in role:
                            std_role = "assistant"
                        elif "patient" in role or "user" in role or "client" in role:
                            std_role = "patient"
                        else:
                            continue
                        
                        if isinstance(content, str):
                            print(f"DEBUG: Adding message with role {std_role} from dict")
                            self.conversation_history.append({
                                "role": std_role,
                                "content": content
                            })
        except json.JSONDecodeError as e:
            print(f"Warning: Could not parse conversation as JSON: {e}")
            # If JSON parsing fails completely, try text-based extraction
            self._extract_conversation_from_text(conversation_text)
            # Extract responses from the conversation history after text extraction
            if self.conversation_history:
                self._extract_responses_from_conversation_history()
        
        if not self.responses and self.questions:
            print("[DEBUG] Using EPDS questions with empty responses")
            for question in self.questions:
                self.responses.append((question, ""))
    
    def _score_response(self, response, question_idx):
        """Map a response to an EPDS score (0–3) using LLM and RAG with provided EPDS guidelines."""
        rag_context = ""
        if self.rag_engine:
            rag_query = f"Edinburgh Postnatal Depression Scale scoring guidelines for question: {self.questions[question_idx]}"
            rag_result = self.rag_engine.retrieve(rag_query, top_k=3)
            if isinstance(rag_result, dict) and "content_list" in rag_result:
                documents = rag_result.get("documents", [])
                filtered_content = []
                for i, doc in enumerate(documents):
                    doc_id = doc.get("title", "") + "|" + doc.get("highlight", "")[:50]
                    if doc_id not in self.seen_documents:
                        self.seen_documents.add(doc_id)
                        if i < len(rag_result["content_list"]):
                            filtered_content.append(rag_result["content_list"][i])
                if filtered_content:
                    rag_context = "\n\n".join(filtered_content)
        
        epds_guidelines = """
        The Edinburgh Postnatal Depression Scale (EPDS) Scoring Guidelines:
        - Items 1, 2, 4: Score 0-3 (top option = 0, bottom = 3).
        1. I have been able to laugh and see the funny side of things:
            - As much as I always could (0)
            - Not quite as much now (1)
            - Definitely not so much now (2)
            - Not at all (3)
        2. I have looked forward with enjoyment to things:
            - As much as I ever did (0)
            - Rather less than I used to (1)
            - Definitely less than I used to (2)
            - Hardly at all (3)
        4. I have been anxious or worried for no good reason:
            - No, not at all (0)
            - Hardly ever (1)
            - Yes, sometimes (2)
            - Yes, very often (3)
        - Items 3, 5-10: Reverse-scored (top = 3, bottom = 0).
        3. I have blamed myself unnecessarily when things went wrong:
            - Yes, most of the time (3)
            - Yes, some of the time (2)
            - Not very often (1)
            - No, never (0)
        5. I have felt scared or panicky for no very good reason:
            - Yes, quite a lot (3)
            - Yes, sometimes (2)
            - No, not much (1)
            - No, not at all (0)
        6. Things have been getting on top of me:
            - Yes, most of the time I haven't been able to cope at all (3)
            - Yes, sometimes I haven't been coping as well as usual (2)
            - No, most of the time I have coped quite well (1)
            - No, I have been coping as well as ever (0)
        7. I have been so unhappy that I have had difficulty sleeping:
            - Yes, most of the time (3)
            - Yes, sometimes (2)
            - Not very often (1)
            - No, not at all (0)
        8. I have felt sad or miserable:
            - Yes, most of the time (3)
            - Yes, quite often (2)
            - Not very often (1)
            - No, not at all (0)
        9. I have been so unhappy that I have been crying:
            - Yes, most of the time (3)
            - Yes, quite often (2)
            - Only occasionally (1)
            - No, never (0)
        10. The thought of harming myself has occurred to me:
            - Yes, quite often (3)
            - Sometimes (2)
            - Hardly ever (1)
            - Never (0)
        """
        
        prompt = f"""
        Respond ONLY with a valid JSON object. No other text, no explanations, no markdown. Just the JSON.
        You are a mental health professional scoring a patient's response to an EPDS question.
        
        Question: {self.questions[question_idx]}
        Patient Response: {response}
        Scoring Guidelines: {epds_guidelines}
        Additional Context: {rag_context}
        
        INSTRUCTIONS:
        - Analyze the response semantically to assign a score (0–3) based on the EPDS scoring guidelines.
        - For questions 1, 2, 4: Top option = 0, bottom = 3.
        - For questions 3, 5–10: Top option = 3, bottom = 0.
        - Return a JSON object with:
        - score: Integer (0–3)
        - explanation: Brief explanation of the scoring decision
        - warning: String (only for question 10 if score ≥1, otherwise empty)
        """
        temp_conversation = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt}
        ]
        result = self.client.chat(self.model, temp_conversation)
        response_text = result['response']
        
        # Thêm phần clean JSON (copy từ extract_and_clean_json trong full_conversation_agent.py)
        cleaned_text = response_text.strip()
        cleaned_text = re.sub(r'^```json\s*', '', cleaned_text)
        cleaned_text = re.sub(r'^```\s*', '', cleaned_text)
        cleaned_text = re.sub(r'\s*```$', '', cleaned_text)
        array_match = re.search(r'$$ .* $$', cleaned_text, re.DOTALL)  # Nếu cần, nhưng ở đây là object nên bỏ
        if array_match:
            cleaned_text = array_match.group(0)
        cleaned_text = re.sub(r'[\x00-\x1F\x7F]', '', cleaned_text)
        cleaned_text = cleaned_text.replace('\u201c', '"').replace('\u201d', '"')
        cleaned_text = cleaned_text.replace('\u2018', "'").replace('\u2019', "'")
        cleaned_text = re.sub(r'([{,])\s*\'([^\']+)\'\s*:', r'\1"\2":', cleaned_text)
        cleaned_text = re.sub(r':\s*\'([^\']+)\'\s*([,}])', r':"\1"\2', cleaned_text)
        
        # Giữ nguyên phần try-except parse
        try:
            result_json = json.loads(cleaned_text)
            if not isinstance(result_json, dict) or 'score' not in result_json:
                raise ValueError("Invalid JSON structure: missing 'score' key")
            print(f"[DEBUG] Successfully parsed score: {result_json['score']}")
            return result_json['score']  # Nếu cần explanation/warning, return full dict
        except (json.JSONDecodeError, ValueError) as e:
            print(f"[DEBUG] Failed to parse LLM response: {response_text}")
            print(f"[DEBUG] Parsing error: {str(e)}")
            # Fallback: Extract score from text if possible
            score_match = re.search(r'"score"\s*:\s*(\d)', response_text)
            if score_match:
                score = int(score_match.group(1))
                print(f"[DEBUG] Extracted score {score} from text")
                return score
            print("[DEBUG] Using default score due to parsing failure")
            return 1  # Default score

    def generate_screening(self):
        """
        Generate a screening based on the patient's responses.
        
        Returns:
            dict: Screening from the assistant with RAG usage information
        """
        epds_score = 0
        score_explanations = []
        warnings = []
        for idx, (question, response) in enumerate(self.responses):
            score = self._score_response(response, idx)
            epds_score += score
            score_explanations.append(f"Q{idx+1}: {question}\nResponse: {response}\nScore: {score}\n")
            if idx == 9 and score >= 1:  # Item 10
                warnings.append("Immediate follow-up required for self-harm risk.")
        
        risk_level = "Low risk" if epds_score <= 9 else "Moderate risk" if epds_score <= 12 else "High risk"
        
        observations = self._summarize_observations()
        print(f"[DEBUG] Generated clinical observations: {observations[:100]}...")
        
        # Create a prompt for diagnosis that includes the observations
        diagnosis_prompt = f"""
        Based on the questionnaire responses, please provide a comprehensive screening assessment using the Edinburgh Postnatal Depression Scale (EPDS).

        Questionnaire responses and Scores:
        {''.join(score_explanations)}
        Warnings: {''.join(warnings) or 'None'}

        Clinical observations and potential concerns:
        {observations}
        EPDS Score: {epds_score}

        IMPORTANT SCREENING CONSIDERATIONS:
        - The EPDS is a screening tool, not a diagnostic instrument.
        - Interpret results as indicators of risk level only.
        - Recommend follow-up based on score thresholds: 0–9 (low risk), 10–12 (moderate risk), 13–30 (high risk).
        - If Item 10 score ≥1, flag for immediate self-harm risk assessment.
        - Avoid definitive diagnoses; emphasize need for professional evaluation.

        Please provide a professional assessment that MUST follow this EXACT structure:

        1. First paragraph: Write a compassionate summary of what you've heard from the patient, showing empathy for their situation.

        2. After that, include a section with the heading "**EPDS Screening Results:**" (exactly as shown, with the asterisks)
        - On the same line, immediately after the heading, state the total score and risk level (e.g., "Total Score: 15/30 - High Risk").
        - Do not add extra newlines between the heading and the results.

        3. Next, include a section with the heading "**Reasoning:**" (exactly as shown, with the asterisks)
        - Immediately after this heading, explain the score calculation and how responses align with EPDS criteria.
        - Do not add extra newlines between the heading and your explanation.

        4. Finally, include a section with the heading "**Recommended Next Steps:**" (exactly as shown, with the asterisks)
        - List specific numbered recommendations (1., 2., 3., etc.), such as referral to a clinician, support resources, or safety planning.
        - Make each recommendation clear and actionable.

        When writing your assessment, use these special tags:
        - Wrap medical terms and conditions in <med>medical term</med> tags
        - Wrap symptoms in <sym>symptom</sym> tags
        - Wrap patient quotes or paraphrases in <quote>patient quote</quote> tags

        EXTREMELY IMPORTANT:
        1. Do NOT include any introductory statements answering the prompt.
        2. Do NOT begin with phrases like "Okay, here's a clinical assessment..."
        3. Start DIRECTLY with the compassionate summary paragraph without any preamble.
        4. Never include meta-commentary about what you're about to write.
        5. Include all four components in the exact order specified.
        6. Format section headings consistently with double asterisks.
        7. Maintain proper spacing between sections (one blank line).
        8. Do not add extra newlines within sections.
        9. Always wrap medical terms, symptoms, and quotes in the specified tags.
        10. Do not provide a formal diagnosis; frame everything as screening results.

        Keep your tone professional but warm, showing empathy while maintaining clinical objectivity.
        """
        
        # Initialize RAG usage information
        rag_usage = None
        if self.rag_engine and not self.disable_rag_evaluation:
            print("[DEBUG] Using RAG for diagnosis...")
            rag_query = f"postnatal depression diagnosis for patient with EPDS score {epds_score}: {observations}"
            rag_result = self.rag_engine.retrieve(rag_query, top_k=5)
            
            if isinstance(rag_result, dict) and "content_list" in rag_result:
                # New RAG format
                documents = rag_result.get("documents", [])
                filtered_content = []
                newly_accessed_docs = []
                
                # Only use documents we haven't seen before
                for i, doc in enumerate(documents):
                    doc_id = doc.get("title", "") + "|" + doc.get("highlight", "")[:50]
                    if doc_id not in self.seen_documents:
                        self.seen_documents.add(doc_id)
                        # Add content only if it's new
                        if i < len(rag_result["content_list"]):
                            filtered_content.append(rag_result["content_list"][i])
                        newly_accessed_docs.append(doc)
                
                if filtered_content:
                    print(f"[DEBUG] Found {len(filtered_content)} new relevant documents for diagnosis")
                    # Don't include raw context in the prompt, instead use system message
                    self.conversation_history.append({
                        "role": "system", 
                        "content": f"Use this additional reference information to help inform your diagnosis, but don't include raw reference text in your response: {' '.join(filtered_content)}"
                    })
                    
                    # Track RAG usage information for new documents only
                    rag_usage = {
                        "documents": newly_accessed_docs,
                        "stats": rag_result.get("stats", {}),
                        "count": len(newly_accessed_docs)
                    }
            else:
                # Legacy format
                context = rag_result
                if context:
                    print(f"[DEBUG] Found {len(context)} relevant documents for diagnosis")
                    
                    # Don't include raw context in the prompt, instead use system message
                    self.conversation_history.append({
                        "role": "system", 
                        "content": f"Use this additional reference information to help inform your diagnosis, but don't include raw reference text in your response: {' '.join(context)}"
                    })
                    
                    # Track RAG usage information
                    if hasattr(self.rag_engine, 'get_accessed_documents'):
                        accessed_docs = self.rag_engine.get_accessed_documents()
                        if accessed_docs:
                            print(f"[DEBUG] RAG used: captured {len(accessed_docs)} relevant documents for diagnosis")
                            rag_usage = {
                                "accessed_documents": accessed_docs,
                                "count": len(accessed_docs)
                            }
                            # Clear the accessed documents for next query
                            self.rag_engine.clear_accessed_documents()
        else:
            # Skip RAG if it's disabled
            if self.disable_rag_evaluation:
                print("[DEBUG] RAG evaluation disabled for diagnosis generation")
        
        self.conversation_history.append({"role": "user", "content": diagnosis_prompt})
        
        result = self.client.chat(self.model, self.conversation_history)
        diagnosis = result['response']
        self.context = result.get('context')
        
        # Add diagnosis to conversation history
        self.conversation_history.append({"role": "assistant", "content": diagnosis})
        
        # Return both the diagnosis and RAG usage information
        return {
            "content": diagnosis,
            "rag_usage": rag_usage
        }
    
    def _summarize_observations(self) -> str:
        """
        Summarize clinical observations from patient responses.
        
        Returns:
            str: Clinical observations and potential concerns
        """
        # Create formatted responses for the summarization
        formatted_responses = self._format_responses()
        
        # Create a prompt for the observation summarization
        summarization_prompt = f"""
        You are a postpartum mental health professional reviewing patient responses to a questionnaire.
        
        Here are the patient's responses:
        {formatted_responses}
        
        Based on these responses, please:
        1. Identify the main symptoms and concerns
        2. Note patterns in the patient's responses
        3. List potential areas of clinical significance
        4. Highlight any risk factors or warning signs
        5. Summarize your observations in clinical language
        
        Format your response as a concise clinical observation summary using professional terminology.
        Focus on extracting the most relevant clinical information while avoiding speculation.
        """
        
        # Create a temporary conversation for generating the observations
        temp_conversation = [
            {"role": "system", "content": "You are a clinical postpartum mental health professional conducting an assessment."},
            {"role": "user", "content": summarization_prompt}
        ]
        
        # Generate the clinical observations using the LLM
        result = self.client.chat(self.model, temp_conversation)
        observations = result['response']
        
        return observations

    def _extract_symptoms_for_query(self):
        """Extract key symptoms from patient responses to create a better RAG query."""
        # This method is kept for backwards compatibility
        # The preferred approach is now to use _summarize_observations() for RAG queries
        symptoms = []
        for question, response in self.responses:
            # Add both question and response to get context
            symptoms.append(f"{question} {response}")
        
        # Join all symptoms into one query string
        combined = " ".join(symptoms)
        
        # Include common mental health terminology to improve RAG retrieval
        query = f"mental health assessment for patient with symptoms: {combined}"
        return query

    def _format_responses(self):
        """Format the patient's responses for diagnosis."""
        formatted = ""
        for i, (question, response) in enumerate(self.responses, 1):
            score = self._score_response(response, i-1)
            formatted += f"Q{i}: {question}\nA{i}: {response}\nScore: {score}\n\n"
        return formatted

    def _extract_conversation_from_text(self, text):
        """Extract conversation turns from plain text when JSON parsing fails."""
        import re
        print("DEBUG: Attempting text-based conversation extraction")
        
        # Define patterns to identify speaker turns
        patterns = [
            # Look for patterns like "Assistant: message" or "Patient: message"
            r'(?:^|\n)(assistant|patient|clinician|therapist|doctor|user|client):\s*(.*?)(?=\n(?:assistant|patient|clinician|therapist|doctor|user|client):|$)',
            # Alternative pattern with quotes or brackets
            r'(?:^|\n)[\'"]?(assistant|patient|clinician|therapist|doctor|user|client)[\'"]?\s*[:\-]\s*[\'"]?(.*?)[\'"]?(?=\n|$)',
            # JSON-like format without proper syntax
            r'role[\'"]?\s*:\s*[\'"]?(assistant|patient|clinician|therapist|doctor|user|client)[\'"]?[,\s]+[\'"]?content[\'"]?\s*:\s*[\'"]?(.*?)[\'"]?(?=[,\}]|$)'
        ]
        
        # Try each pattern until we get some results
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE | re.DOTALL)
            if matches:
                print(f"DEBUG: Found {len(matches)} conversation turns with pattern")
                
                for role_text, content in matches:
                    # Standardize role
                    role = role_text.lower()
                    if role in ["assistant", "clinician", "therapist", "doctor"]:
                        std_role = "assistant"
                    else:
                        std_role = "patient"
                    
                    # Clean up content
                    content = content.strip()
                    if content:
                        self.conversation_history.append({
                            "role": std_role,
                            "content": content
                        })
                
                print(f"DEBUG: Extracted {len(self.conversation_history)} messages from text")
                return
        
        # Last resort: just treat the whole thing as a single assistant message
        if not self.conversation_history:
            print("DEBUG: No patterns matched, treating entire text as assistant message")
            self.conversation_history.append({
                "role": "assistant",
                "content": text
            })
            
    def _extract_responses_from_conversation_history(self):
        """
        Extract question-answer pairs directly from the parsed conversation history.
        This method is called after successful JSON parsing to populate the responses.
        """
        # Clear existing responses
        self.responses = []
        
        print("DEBUG: Extracting responses from conversation history...")
        
        current_question = None
        
        # Process conversation history to extract Q&A pairs
        for i in range(len(self.conversation_history)):
            msg = self.conversation_history[i]
            
            if msg["role"] == "assistant":
                # This is a potential question/prompt from the assistant
                current_question = msg["content"]
            elif msg["role"] == "patient" and current_question:
                # This is an answer to the previous question/prompt
                self.responses.append((current_question, msg["content"]))
                current_question = None
        
        print(f"DEBUG: Extracted {len(self.responses)} Q&A pairs from conversation history")