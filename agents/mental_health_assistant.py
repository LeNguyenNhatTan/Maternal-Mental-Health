import os
import re
import json
from utils.llm_client_base import LLMClient
from utils.rag_engine import RAGEngine

class MentalHealthAssistant:
    def __init__(self, provider="ollama", provider_options=None, model="qwen3:4b", 
                 questions=None, rag_engine=None, questionnaire_name=None):
        """
        Initialize the Mental Health Assistant agent.
        
        Args:
            provider (str): LLM provider to use (ollama or groq)
            provider_options (dict, optional): Options to pass to the provider client
            model (str): Model to use with the provider
            questions (list): List of questions from the questionnaire
            rag_engine (RAGEngine, optional): RAG engine for document retrieval
            questionnaire_name (str, optional): Name of the questionnaire being used
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
        self.questions = questions
        self.responses = []
        self.current_question_idx = 0
        self.context = None
        self.rag_engine = rag_engine
        self.has_introduced = False  # Flag to track if introduction has been given
        self.questionnaire_name = questionnaire_name
        
        self.seen_documents = set()

        # Load system prompt from file
        prompt_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                                  "prompts", "mental_health_assistant_prompt.txt")
        if os.path.exists(prompt_path):
            with open(prompt_path, 'r') as f:
                self.system_prompt = f.read()
        else:
            self.system_prompt = """
            You are a professional mental health clinician conducting a postnatal depression screening using the Edinburgh Postnatal Depression Scale (EPDS).
            
            INSTRUCTIONS:
            - Use ONLY the EPDS questionnaire with its 10 questions.
            - Administer questions in a warm, empathetic, and professional manner.
            - Do NOT make diagnostic assumptions before completing all questions.
            - After collecting responses, analyze them semantically to assign EPDS scores (0–3) per question, using the following guidelines:
                - Questions 1-2 (positive emotions): 0 (Always/Often), 1 (Sometimes), 2 (Rarely), 3 (Never)
                - Questions 3-10 (negative symptoms): 0 (Never), 1 (Sometimes), 2 (Often), 3 (Always)
            - Calculate the total EPDS score (0–30) and determine the risk level:
                - 0–9: Low risk
                - 10–12: Moderate risk
                - 13–30: High risk
            - Provide a detailed rationale linking responses to EPDS criteria.
            - Use tags: <med>medical term</med>, <sym>symptom</sym>, <quote>patient quote</quote>.
            """
        self.conversation_history = [{"role": "system", "content": self.system_prompt}]
    
    
    
    def get_next_message(self, patient_response=None):
        """
        Get the next message from the assistant.
        
        Args:
            patient_response (str): Response from the patient
            
        Returns:
            str or dict: Next message from the assistant, possibly with RAG info
        """
        if patient_response:
            # Add patient's response to conversation history
            self.conversation_history.append({"role": "user", "content": patient_response})
            if self.current_question_idx > 0:  # Only add to responses if we've asked a question
                self.responses.append((self.questions[self.current_question_idx-1], patient_response))
        
        # If this is the first interaction, provide an introduction
        if not self.has_introduced:
            self.has_introduced = True
            
            # Generate introduction based on the questionnaire content directly
            intro_message = self._generate_introduction()
            
            # Add introduction to conversation history
            self.conversation_history.append({"role": "assistant", "content": intro_message})
            return intro_message
        
        if self.current_question_idx < len(self.questions):
            next_question = self.questions[self.current_question_idx]
            print(f"[DEBUG] Asking EPDS question #{self.current_question_idx + 1}: {next_question[:50]}...")
            self.current_question_idx += 1
            self.conversation_history.append({"role": "assistant", "content": next_question})
            return next_question
        else:
            print("[DEBUG] All EPDS questions asked. Generating diagnosis.")
            return self.generate_diagnosis()
    
    def _generate_introduction(self):
        """Generate an introduction for the mental health assessment."""
        print("[DEBUG] Retrieving full questionnaire document for introduction...")
        
        # Get the full questionnaire document if available
        full_questionnaire_content = ""
        if self.questionnaire_name and self.rag_engine:
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
        else:
            full_questionnaire_content = ""
        
        intro_prompt = f"""
        You are a postpartum professional mental health clinician about to conduct an assessment using a mental health questionnaire.
        
        Here is the full questionnaire document you will be administering:
        
        ```
        {full_questionnaire_content or chr(10).join([f"{i+1}. {q}" for i, q in enumerate(self.questions)])}
        ```
        
        IMPORTANT INSTRUCTIONS:
        - Use ONLY the actual name of the questionnaire as shown in the document above
        - DO NOT introduce this as a "Somatic Symptom Disorder questionnaire" unless that is explicitly the name in the document
        - DO NOT assume what specific condition is being assessed
        - The questionnaire may be a general mental health assessment or focused on various conditions
        - Your role is to administer the questionnaire without making diagnostic assumptions up front
        
        Based on this questionnaire document, please generate a warm, professional introduction to the patient that:
        1. Introduces yourself as a postpartum mental health professional
        2. Identifies the specific questionnaire you're using by name (from the document)
        3. Explains the purpose of this specific assessment 
        4. Reassures the patient about confidentiality and creating a safe space
        5. Briefly explains how the assessment will proceed ({len(self.questions)} questions)
        6. Indicates you're ready to begin with the first question
        
        Keep your tone professional but warm, showing empathy while maintaining clinical objectivity.
        Make sure to correctly identify and name the specific questionnaire you're administering.
        For the first interaction, provide a complete introduction followed by your first question.
        
        This is real-time conversation with a human patient, so make your introduction engaging, natural, and conversational.
        """
        
        # Create a temporary conversation for generating the introduction
        temp_conversation = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": intro_prompt}
        ]
        result = self.client.chat(self.model, temp_conversation)
        return result['response']
    
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
        
        epds_guidelines = [
            {
                "question": "I have been able to laugh and see the funny side of things",
                "reverse": False,  # top=0, bottom=3
                "options": [
                    {"text": "As much as I always could", "score": 0},
                    {"text": "Not quite as much now", "score": 1},
                    {"text": "Definitely not so much now", "score": 2},
                    {"text": "Not at all", "score": 3},
                ],
            },
            {
                "question": "I have looked forward with enjoyment to things",
                "reverse": False,
                "options": [
                    {"text": "As much as I ever did", "score": 0},
                    {"text": "Rather less than I used to", "score": 1},
                    {"text": "Definitely less than I used to", "score": 2},
                    {"text": "Hardly at all", "score": 3},
                ],
            },
            {
                "question": "I have blamed myself unnecessarily when things went wrong",
                "reverse": True,  
                "options": [
                    {"text": "Yes, most of the time", "score": 3},
                    {"text": "Yes, some of the time", "score": 2},
                    {"text": "Not very often", "score": 1},
                    {"text": "No, never", "score": 0},
                ],
            },
            {
                "question": "I have been anxious or worried for no good reason",
                "reverse": False,
                "options": [
                    {"text": "No, not at all", "score": 0},
                    {"text": "Hardly ever", "score": 1},
                    {"text": "Yes, sometimes", "score": 2},
                    {"text": "Yes, very often", "score": 3},
                ],
            },
            {
                "question": "I have felt scared or panicky for no very good reason",
                "reverse": True,
                "options": [
                    {"text": "Yes, quite a lot", "score": 3},
                    {"text": "Yes, sometimes", "score": 2},
                    {"text": "No, not much", "score": 1},
                    {"text": "No, not at all", "score": 0},
                ],
            },
            {
                "question": "Things have been getting on top of me",
                "reverse": True,
                "options": [
                    {"text": "Yes, most of the time I haven't been able to cope at all", "score": 3},
                    {"text": "Yes, sometimes I haven't been coping as well as usual", "score": 2},
                    {"text": "No, most of the time I have coped quite well", "score": 1},
                    {"text": "No, I have been coping as well as ever", "score": 0},
                ],
            },
            {
                "question": "I have been so unhappy that I have had difficulty sleeping",
                "reverse": True,
                "options": [
                    {"text": "Yes, most of the time", "score": 3},
                    {"text": "Yes, sometimes", "score": 2},
                    {"text": "Not very often", "score": 1},
                    {"text": "No, not at all", "score": 0},
                ],
            },
            {
                "question": "I have felt sad or miserable",
                "reverse": True,
                "options": [
                    {"text": "Yes, most of the time", "score": 3},
                    {"text": "Yes, quite often", "score": 2},
                    {"text": "Not very often", "score": 1},
                    {"text": "No, not at all", "score": 0},
                ],
            },
            {
                "question": "I have been so unhappy that I have been crying",
                "reverse": True,
                "options": [
                    {"text": "Yes, most of the time", "score": 3},
                    {"text": "Yes, quite often", "score": 2},
                    {"text": "Only occasionally", "score": 1},
                    {"text": "No, never", "score": 0},
                ],
            },
            {
                "question": "The thought of harming myself has occurred to me",
                "reverse": True,
                "options": [
                    {"text": "Yes, quite often", "score": 3},
                    {"text": "Sometimes", "score": 2},
                    {"text": "Hardly ever", "score": 1},
                    {"text": "Never", "score": 0},
                ],
            },
        ]

        
        prompt = f"""
        Respond ONLY with a valid JSON object. 
        No explanations, no markdown, no text outside JSON. 

        You are a licensed mental health clinician scoring a patient's response to the Edinburgh Postnatal Depression Scale (EPDS).

        TASK:
        - You will receive one EPDS question, the patient’s free-text response, and the official EPDS scoring guidelines for that question. 
        - Your job is to evaluate the meaning of the patient’s response and map it to the closest guideline option. 
        - Then return the corresponding score, along with a short explanation of your reasoning.

        IMPORTANT RULES:
        1. Use SEMANTIC matching, not exact word matching. For example:
        - If patient says "I can still laugh but not as much as before", it matches "Not quite as much now".
        - If patient says "I never enjoy things anymore", it matches "Hardly at all".
        2. Return JSON with exactly three fields:
        - score: Integer (0–3)
        - explanation: Short string (why the response matches the selected option)
        - warning: 
            - For Question 10, if score ≥ 1 → "Possible self-harm risk, alert clinician."
            - Otherwise → empty string "".
        3. Do not invent new options. Only use the ones in EPDS Guidelines.
        4. Do not include any other commentary, markdown, or formatting. 
        Output must be pure JSON only.

        DATA PROVIDED:
        EPDS Question: {self.questions[question_idx]}
        Patient Response: {response}
        EPDS Guidelines: {json.dumps(epds_guidelines[question_idx], ensure_ascii=False)}

        VALID EXAMPLE OUTPUT:
        {{
        "score": 2,
        "explanation": "Patient reports enjoying things less than before, which matches 'Definitely less than I used to'.",
        "warning": ""
        }}
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

    def generate_diagnosis(self):
        """
        Generate a diagnosis based on the patient's responses.
        
        Returns:
            dict: Diagnosis from the assistant with RAG usage information
        """
        epds_score = 0
        score_explanations = []
        for idx, (question, response) in enumerate(self.responses):
            score = self._score_response(response, idx)
            epds_score += score
            score_explanations.append(f"Q{idx+1}: {question}\nResponse: {response}\nScore: {score}\n")
        
        risk_level = "Low risk" if epds_score <= 9 else "Moderate risk" if epds_score <= 12 else "High risk"        
        observations = self._summarize_observations()
        # print(f"[DEBUG] Generated clinical observations: {observations[:100]}...")
        
        # Create a prompt for diagnosis that includes the observations
        diagnosis_prompt = f"""
        Respond ONLY with a valid JSON object. No other text, no explanations, no markdown.
        
        Your JSON must have exactly these keys:
        - "summary": string
        - "screening_impression": string (include "Total EPDS Score: {epds_score}, Risk Level: {risk_level}, concerns...")
        - "reasoning": string
        - "next_steps": list of strings
        - "rag_info": object (can be empty)

        Questionnaire responses and scoring:
        {''.join(score_explanations)}
        
        Clinical observations and potential concerns:
        {observations}
        
        Total EPDS Score: {epds_score}
        Risk Level: {risk_level}
        
        IMPORTANT SCREENING CONSIDERATIONS:
        - This is a screening tool, not a diagnostic evaluation
        - Identify potential levels of risk for <med>postpartum depression</med> or related concerns
        - Be open to a range of possibilities including <med>anxiety</med>, <med>adjustment difficulties</med>, and <med>stress-related symptoms</med>
        - Do not assign a formal diagnosis; instead describe screening impressions and level of concern
        - If responses are inconclusive, clearly indicate uncertainty and need for further evaluation
        
        Please analyze these responses and observations and provide a professional screening assessment that MUST follow this EXACT structure:

        1. First paragraph: Write a compassionate summary of what you've heard from the patient, showing empathy for their postpartum experience.
        
        2. After that, include a section with the heading "**Screening Impression:**" (exactly as shown, with the asterisks)
        - On the same line, immediately after the heading, provide the screening impression in this format: "Total EPDS Score: {epds_score}, Risk Level: {risk_level} (e.g., concerns include <sym>symptom1</sym>, <sym>symptom2</sym>)"
        - Do not add extra newlines between the heading and the impression
        
        3. Next, include a section with the heading "**Reasoning:**" (exactly as shown, with the asterisks)
        - Immediately after this heading, explain your rationale for the screening impression, referencing symptoms and EPDS guidelines
        - Do not add extra newlines between the heading and your explanation
        
        4. Finally, include a section with the heading "**Recommended Next Steps/Support Options:**" (exactly as shown, with the asterisks)
        - List specific numbered recommendations (1., 2., 3., etc.)
        - Make each recommendation clear, supportive, and actionable
        
        When writing your screening assessment, use these special tags:
        - Wrap medical terms and conditions in <med>medical term</med> tags
        - Wrap symptoms in <sym>symptom</sym> tags
        - Wrap patient quotes or paraphrases in <quote>patient quote</quote> tags
        
        EXTREMELY IMPORTANT:
        1. Return a JSON object with keys: summary, screening_impression, reasoning, next_steps, rag_info (if applicable)
        2. Do NOT include any introductory statements answering the prompt
        3. Do NOT begin with phrases like "Okay, here's a clinical assessment..."
        4. Start DIRECTLY with the compassionate summary paragraph without any preamble
        5. Never include meta-commentary about what you're about to write
        6. Include all four components in the exact order specified
        7. Format section headings consistently with double asterisks
        8. Maintain proper spacing between sections (one blank line)
        9. Do not add extra newlines within sections
        10. Do not repeat any sections or include incomplete sentences
        11. Always wrap medical terms, symptoms, and quotes in the specified tags

        Keep your tone professional but warm, showing empathy while maintaining clinical objectivity.
        """
        
        # Initialize RAG usage information
        rag_usage = None
        if self.rag_engine:
            print("[DEBUG] Using RAG for screening...")
            rag_query = f"postpartum depression screening impression for patient with EPDS score {epds_score}: {observations}"
            rag_result = self.rag_engine.retrieve(rag_query, top_k=5)
            
            if isinstance(rag_result, dict) and "content_list" in rag_result:
                documents = rag_result.get("documents", [])
                filtered_content = []
                newly_accessed_docs = []
                
                for i, doc in enumerate(documents):
                    doc_id = doc.get("title", "") + "|" + doc.get("highlight", "")[:50]
                    if doc_id not in self.seen_documents:
                        self.seen_documents.add(doc_id)
                        if i < len(rag_result["content_list"]):
                            filtered_content.append(rag_result["content_list"][i])
                        newly_accessed_docs.append(doc)
                
                if filtered_content:
                    print(f"[DEBUG] Found {len(filtered_content)} new relevant documents for diagnosis")
                    self.conversation_history.append({
                        "role": "system", 
                        "content": f"Use this additional reference information to help inform your postpartum depression screening impression, but don't include raw reference text in your response: {' '.join(filtered_content)}"
                    })
                    rag_usage = {
                        "documents": newly_accessed_docs,
                        "stats": rag_result.get("stats", {}),
                        "count": len(newly_accessed_docs)
                    }
        
        self.conversation_history.append({"role": "user", "content": diagnosis_prompt})
        
        result = self.client.chat(self.model, self.conversation_history)
        diagnosis = result['response']
        self.context = result['context']
        
        try:
            # Làm sạch JSON
            cleaned_text = diagnosis.strip()
            cleaned_text = re.sub(r'^```json\s*', '', cleaned_text)
            cleaned_text = re.sub(r'^```\s*', '', cleaned_text)
            cleaned_text = re.sub(r'\s*```$', '', cleaned_text)
            cleaned_text = re.sub(r'[\x00-\x1F\x7F]', '', cleaned_text)
            cleaned_text = cleaned_text.replace('\u201c', '"').replace('\u201d', '"')
            cleaned_text = cleaned_text.replace('\u2018', "'").replace('\u2019', "'")
            cleaned_text = re.sub(r'([{,])\s*\'([^\']+)\'\s*:', r'\1"\2":', cleaned_text)
            cleaned_text = re.sub(r':\s*\'([^\']+)\'\s*([,}])', r':"\1"\2', cleaned_text)
            
            # Thêm bước kiểm tra JSON hợp lệ
            try:
                result_json = json.loads(cleaned_text, strict=False)
            except json.JSONDecodeError as e:
                print(f"[DEBUG] Initial JSON parsing failed: {str(e)}")
                # Thử làm sạch thêm nếu có lỗi liên quan đến next_steps
                cleaned_text = re.sub(r'"\*\*Recommended Next Steps/Support Options:\*\*\n([\s\S]*?)"', r'[\1]', cleaned_text)
                cleaned_text = re.sub(r'\n\d+\.\s*', '","', cleaned_text)
                cleaned_text = re.sub(r'^$$ \s*"', '[', cleaned_text)
                cleaned_text = re.sub(r'"\s* $$$', ']', cleaned_text)
                result_json = json.loads(cleaned_text, strict=False)
            
            # Kiểm tra cấu trúc JSON
            required_keys = {"summary", "screening_impression", "reasoning", "next_steps", "rag_info"}
            if not all(key in result_json for key in required_keys):
                raise ValueError("Missing required keys in JSON response")
            
            # Định dạng screening_impression
            screening_impression = result_json['screening_impression']
            if isinstance(screening_impression, dict) and "concerns" in screening_impression:
                concerns_text = f"({', '.join(screening_impression['concerns'])})"
            else:
                concerns_text = f"Total EPDS Score: {epds_score}, Risk Level: {risk_level} ({screening_impression})"
            
            # Định dạng next_steps
            next_steps = result_json['next_steps']
            if isinstance(next_steps, str):
                next_steps = [step.strip() for step in next_steps.split('\n') if step.strip().startswith(('\d.', '**'))]
                next_steps = [re.sub(r'^\d+\.\s*', '', step) for step in next_steps]
            
            formatted_diagnosis = (
                f"{result_json['summary']}\n\n"
                f"**Screening Impression:** {concerns_text}\n\n"
                f"**Reasoning:** {result_json['reasoning']}\n\n"
                f"**Recommended Next Steps/Support Options:**\n"
                f"{'\n'.join(f'{i+1}. {step}' for i, step in enumerate(next_steps))}\n"
            )
            
            if result_json.get('rag_info'):
                formatted_diagnosis += f"\nRAG Info: {json.dumps(result_json['rag_info'])}\n"
            
            # Loại bỏ các dòng trùng lặp
            lines = formatted_diagnosis.split('\n')
            seen_lines = set()
            cleaned_lines = []
            for line in lines:
                if line.strip() and line not in seen_lines and not line.endswith('...'):
                    cleaned_lines.append(line)
                    seen_lines.add(line)
            formatted_diagnosis = '\n'.join(cleaned_lines)
            
            self.conversation_history.append({"role": "assistant", "content": formatted_diagnosis})
            
            return {
                "content": formatted_diagnosis,
                "rag_usage": rag_usage
            }
        except (json.JSONDecodeError, ValueError) as e:
            print(f"[DEBUG] Failed to parse diagnosis: {diagnosis}")
            print(f"[DEBUG] Parsing error: {str(e)}")
            fallback_diagnosis = (
                f"Failed to parse JSON response, but screening completed.\n\n"
                f"**Screening Impression:** Total EPDS Score: {epds_score}, Risk Level: {risk_level}.\n\n"
                f"**Reasoning:** Unable to parse detailed reasoning due to JSON error.\n\n"
                f"**Recommended Next Steps/Support Options:**\n"
                f"1. Consult a <med>mental health professional</med> for further evaluation.\n"
            )
            self.conversation_history.append({"role": "assistant", "content": fallback_diagnosis})
            return {
                "content": fallback_diagnosis,
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
        You are a licensed mental health professional conducting a postpartum depression screening (this is not a formal diagnosis).

        Here are the patient's responses:
        {formatted_responses}

        Your task is to produce a concise professional screening summary.

        INSTRUCTIONS:
        1. Identify the main <sym>symptoms</sym> and emotional/behavioral concerns expressed in the responses.
        2. Note any consistent patterns across answers (e.g., low mood, loss of enjoyment, anxiety, guilt, difficulty coping).
        3. List potential areas of screening significance, particularly those relevant to <med>postpartum depression</med> or <med>anxiety</med>.
        4. Highlight specific risk factors:
        - Persistent sadness, loss of pleasure, crying spells
        - Sleep difficulties linked to mood
        - Feelings of worthlessness or guilt
        - Anxiety, panic, or being overwhelmed
        - Any mention of self-harm or suicidal thoughts → flag clearly as a safety concern
        5. Provide a clear screening observation summary in professional language:
        - Use objective, non-judgmental phrasing
        - Avoid speculation or formal diagnosis
        - Focus on risk screening, not treatment recommendations

        OUTPUT REQUIREMENTS:
        - Write as a concise, structured screening note (like a mental health professional would record in patient documentation).
        - Prioritize clarity and relevance for clinical screening.
        - Do not include extraneous commentary.
        """
        temp_conversation = [
            {"role": "system", "content": "You are a clinical mental health professional conducting a postpartum depression screening (not a formal diagnosis)."},
            {"role": "user", "content": summarization_prompt}
        ]

        # Generate the clinical observations using the LLM
        result = self.client.chat(self.model, temp_conversation)
        return result['response']\
            
    def _format_responses(self):
        """Format the patient's responses for diagnosis."""
        formatted = ""
        for i, (question, response) in enumerate(self.responses, 1):
            score = self._score_response(response, i-1)
            formatted += f"Q{i}: {question}\nA{i}: {response}\nScore: {score}\n\n"
        return formatted

    def respond(self, message, conversation_history=None):
        """
        Generate a response to the patient's message.
        
        Args:
            message: Message from the patient
            conversation_history: Optional conversation history to use
            
        Returns:
            dict: Response with content and RAG usage information
        """
        # Use provided conversation history or default to self.conversation_history
        history = conversation_history or self.conversation_history
        
        # Create a copy to avoid modifying the original
        history_copy = history.copy()
        
        # Add the user message
        history_copy.append({"role": "user", "content": message})
        
        print(f"[DEBUG] Generating response to: {message[:50]}...")
        
        # Query RAG engine if available
        if self.rag_engine:
            print("[DEBUG] Querying RAG engine for relevant context")
            rag_result = self.rag_engine.retrieve(message, top_k=3)
            
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
                    context_str = "\n\n".join(filtered_content)
                    # Add context to the message
                    system_message = {
                        "role": "system", 
                        "content": f"Consider this additional information when responding:\n{context_str}"
                    }
                    history_copy.append(system_message)
                    
                    rag_usage = {
                        "documents": newly_accessed_docs,
                        "stats": rag_result.get("stats", {}),
                        "count": len(newly_accessed_docs)
                    }
            else:
                # Legacy format handling
                context = rag_result
                if context:
                    print(f"[DEBUG] Found {len(context)} relevant documents")
                    context_str = "\n\n".join(context)
                    # Add context to the message
                    system_message = {
                        "role": "system", 
                        "content": f"Consider this additional information when responding:\n{context_str}"
                    }
                    history_copy.append(system_message)
                    
                    rag_usage = {
                        "accessed_documents": [{"title": "Unknown"}] if context else [],
                        "count": 1 if context else 0
                    }
        
        # Generate response
        result = self.client.chat(self.model, history_copy)
        response = result['response']
        print(f"[DEBUG] Generated response: {response[:50]}...")
        
        # If RAG was used, get the accessed documents
        if self.rag_engine and hasattr(self.rag_engine, 'get_accessed_documents'):
            accessed_docs = self.rag_engine.get_accessed_documents()
            if accessed_docs:
                print(f"[DEBUG] RAG used: found {len(accessed_docs)} relevant documents")
                rag_usage = {
                    "accessed_documents": accessed_docs,
                    "count": len(accessed_docs)
                }
                # Clear the accessed documents for next query
                self.rag_engine.clear_accessed_documents()
        
        # Return both the response and RAG usage information
        return {
            "content": response,
            "rag_usage": rag_usage
        }

def get_context_for_question(self, question: str) -> dict:
    """Get relevant context for a question from the RAG engine."""
    # Get context from RAG engine
    context = self.rag_engine.get_context_for_question(question)
    
    # Handle both old format (string/list) and new format (dictionary)
    if isinstance(context, dict) and "content" in context:
        # New format - already has all we need
        rag_usage = {
            "documents": context.get("documents", []),
            "stats": context.get("stats", {}),
        }
        return {
            "content": context["content"],
            "rag_usage": rag_usage
        }
    else:
        # Old format - convert to new format
        rag_usage = {
            "count": 1 if context else 0,
            "accessed_documents": [{"title": "Unknown"}] if context else []
        }
        # If it's a list, join items with newlines
        if isinstance(context, list):
            context = "\n\n".join(context)
            
        return {
            "content": context,
            "rag_usage": rag_usage
        }
