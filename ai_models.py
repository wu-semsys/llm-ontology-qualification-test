from abc import ABC, abstractmethod
from openai import OpenAI, OpenAIError
import google.generativeai as genai
from ollama import Client as OllamaClient
import anthropic
import time
import random
from together import Together
from groq import Groq
from utils import ONTOLOGY_EVALUATION_INSTRUCTION, print_color
from colorama import Fore, Style

class AIModelFactory:
    @staticmethod
    def create_model(model_type, config):
        model_provider, model_name = model_type.split(':') if ':' in model_type else (model_type, None)
        
        if model_provider == 'openai':
            return OpenAIModel(config.OPENAI_API_KEY, model_name or "gpt-4")
        elif model_provider == 'deepseek':
            return DeepSeekModel(config.DEEPSEEK_API_KEY, model_name or "deepseek-coder")
        elif model_provider == 'gemini':
            return GeminiModel(config.GOOGLE_API_KEY, model_name or "gemini-1.5-flash")
        elif model_provider == 'ollama':
            return OllamaModel(config.OLLAMA_HOST, model_name or "llama3")
        elif model_provider == 'anthropic':
            return AnthropicModel(config.ANTHROPIC_API_KEY, model_name or "claude-3-5-sonnet-20240620")
        elif model_provider == 'together':
            return TogetherAIModel(config.TOGETHER_API_KEY, model_name or "Qwen/Qwen2-72B-Instruct")
        elif model_provider == 'groq':
            return GroqModel(config.GROQ_API_KEY, model_name or "llama3-8b-8192")
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

class AIModel(ABC):
    def __init__(self, model_name):
        self.model_name = model_name
        self.system_message = {"role": "system", "content": ONTOLOGY_EVALUATION_INSTRUCTION}
        self.initial_messages = [self.system_message]

    @abstractmethod
    def _generate_response(self, messages):
        pass

    def get_response(self, prompt, max_retries=10, initial_retry_delay=1):
        messages = self.initial_messages + [{"role": "user", "content": prompt}]
        retry_delay = initial_retry_delay
        log = {
            "model": self.model_name,
            "messages": messages,
            "attempts": []
        }
        for attempt in range(max_retries):
            try:
                print_color(f"      Attempt {attempt + 1}/{max_retries}", Fore.YELLOW)
                start_time = time.time()
                response, attempt_log = self._generate_response(messages)
                end_time = time.time()
                
                attempt_log["execution_time"] = end_time - start_time
                log["attempts"].append(attempt_log)
                log["final_response"] = response
                
                print_color(f"      Response received in {end_time - start_time:.2f} seconds", Fore.GREEN)
                return response, log
            except Exception as e:
                error_message = str(e)
                attempt_log = {
                    "attempt": attempt + 1,
                    "error": error_message,
                    "timestamp": time.time()
                }
                log["attempts"].append(attempt_log)
                print_color(f"      Error: {error_message}", Fore.RED)
                
                if "rate_limit" in error_message.lower():
                    if attempt < max_retries - 1:
                        sleep_time = retry_delay * (2 ** attempt) + random.uniform(0, 1)
                        attempt_log["retry_delay"] = sleep_time
                        print_color(f"      Rate limit reached. Retrying in {sleep_time:.2f} seconds...", Fore.YELLOW)
                        time.sleep(sleep_time)
                    else:
                        print_color("      Max retries reached due to rate limiting.", Fore.RED)
                        return "Error: Rate limit exceeded. Please try again later.", log
                else:
                    print_color(f"      Unexpected error. Retrying...", Fore.YELLOW)
        
        print_color("      Max retries reached without successful response.", Fore.RED)
        return "Error: Max retries reached without successful response.", log


class OpenAIModel(AIModel):
    def __init__(self, api_key, model_name="gpt-4"):
        super().__init__(model_name)
        self.client = OpenAI(api_key=api_key)
        self.assistant = self._create_assistant()
        self.thread = self.client.beta.threads.create()

    def _create_assistant(self):
        return self.client.beta.assistants.create(
            name="Ontology Evaluation Assistant",
            instructions=ONTOLOGY_EVALUATION_INSTRUCTION,
            model=self.model_name,
        )

    def _generate_response(self, messages, max_retries=10, retry_delay=5):
        log = {
            "assistant_id": self.assistant.id,
            "thread_id": self.thread.id,
            "steps": []
        }
       
        # Create a message in the thread
        message = self.client.beta.threads.messages.create(
            thread_id=self.thread.id,
            role="user",
            content=messages[-1]["content"]
        )
        log["steps"].append({"action": "create_message", "message_id": message.id})

        # Run the assistant with retry mechanism
        for attempt in range(max_retries):
            try:
                run = self.client.beta.threads.runs.create(
                    thread_id=self.thread.id,
                    assistant_id=self.assistant.id
                )
                log["steps"].append({"action": "create_run", "run_id": run.id, "attempt": attempt + 1})

                # Wait for the run to complete
                failed_count = 0
                while True:
                    run_status = self.client.beta.threads.runs.retrieve(
                        thread_id=self.thread.id,
                        run_id=run.id
                    )
                    print(f"Retrieving Thread {self.thread.id} Run Status: {run_status.status}")
                    log["steps"].append({"action": "check_status", "status": run_status.status, "attempt": attempt + 1})
                   
                    if run_status.status == "completed":
                        # Retrieve and return the assistant's response
                        messages = self.client.beta.threads.messages.list(thread_id=self.thread.id)
                        log["steps"].append({"action": "retrieve_messages", "message_count": len(messages.data)})
                       
                        for message in messages.data:
                            if message.role == "assistant":
                                response = message.content[0].text.value
                                log["steps"].append({"action": "extract_response", "message_id": message.id})
                                return response, log
                       
                        return "No response from assistant.", log
                    elif run_status.status == "failed":
                        failed_count += 1
                        if failed_count >= max_retries:
                            print(f"Run failed {max_retries} times, skipping...")
                            log["steps"].append({"action": "skip_after_failures", "failures": failed_count})
                            return f"Error: Run failed {max_retries} times", log
                        print(f"{Fore.YELLOW}Run failed, waiting for {retry_delay} seconds before retrying...{Fore.RESET}")
                        time.sleep(retry_delay)
                    else:
                        print(f"{Fore.YELLOW}Waiting for 1 second before checking status again...{Fore.RESET}")
                        time.sleep(1)  # Wait before checking status again
           
            except OpenAIError as e:
                print(f"OpenAI API error on attempt {attempt + 1}: {str(e)}")
                log["steps"].append({"action": "api_error", "error": str(e), "attempt": attempt + 1})
                if attempt < max_retries - 1:
                    print(f"{Fore.YELLOW}Waiting for {retry_delay} seconds before retrying...{Fore.RESET}")
                    time.sleep(retry_delay)
                else:
                    return f"Error: OpenAI API error after {max_retries} attempts", log

        return f"Error: Failed to get a response after {max_retries} attempts", log

class DeepSeekModel(AIModel):
    def __init__(self, api_key, model_name="deepseek-chat"):
        super().__init__(model_name)
        self.client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com/v1")

    def _generate_response(self, messages):
        log = {"steps": []}

        # Prepare the messages for DeepSeek's format
        deepseek_messages = [
            {"role": "system", "content": ONTOLOGY_EVALUATION_INSTRUCTION},
            *[{"role": msg["role"], "content": msg["content"]} for msg in messages if msg["role"] != "system"]
        ]

        log["steps"].append({"action": "prepare_messages"})

        # Create a chat completion
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=deepseek_messages,
            temperature=1,
        )

        log["steps"].append({"action": "create_chat_completion", "response_id": response.id})

        return response.choices[0].message.content.strip(), log

class AnthropicModel(AIModel):
    def __init__(self, api_key, model_name="claude-3-sonnet-20240229"):
        super().__init__(model_name)
        self.client = anthropic.Anthropic(api_key=api_key)
    
    def _generate_response(self, messages):
        log = {"steps": []}

        # Prepare the messages for Anthropic's format
        anthropic_messages = [
            {"role": msg["role"], "content": msg["content"]} for msg in messages if msg["role"] != "system"
        ]

        log["steps"].append({"action": "prepare_messages"})

        # Create a message
        response = self.client.messages.create(
            model=self.model_name,
            max_tokens=1000,
            temperature=1,
            system=ONTOLOGY_EVALUATION_INSTRUCTION,
            messages=anthropic_messages
        )

        log["steps"].append({"action": "create_message", "message_id": response.id})

        return response.content[0].text, log

class GeminiModel(AIModel):
    def __init__(self, api_key, model_name="gemini-1.5-flash"):
        super().__init__(model_name)
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)

    def _generate_response(self, messages):
        log = {"steps": []}

        # Prepare the prompt for Gemini
        prompt = "\n".join([msg['content'] for msg in messages])
        log["steps"].append({"action": "prepare_prompt"})

        # Generate content
        response = self.model.generate_content(prompt)
        log["steps"].append({"action": "generate_content"})

        return response.text.strip(), log

class OllamaModel(AIModel):
    def __init__(self, host, model_name='llama3'):
        super().__init__(model_name)
        self.client = OllamaClient(host=host)

    def _generate_response(self, messages):
        log = {"steps": []}

        log["steps"].append({"action": "prepare_messages"})

        # Generate chat response
        response = self.client.chat(model=self.model_name, messages=messages)
        log["steps"].append({"action": "generate_chat_response"})

        return response['message']['content'].strip(), log

class TogetherAIModel(AIModel):
    def __init__(self, api_key, model_name="Qwen/Qwen2-72B-Instruct"):
        super().__init__(model_name)
        self.client = Together(api_key=api_key)
    
    def _generate_response(self, messages):
        log = {"steps": []}
        
        # Prepare the messages for Together AI's format
        together_messages = [
            {"role": "system", "content": ONTOLOGY_EVALUATION_INSTRUCTION},
            *[{"role": msg["role"], "content": msg["content"]} for msg in messages if msg["role"] != "system"]
        ]
        
        log["steps"].append({"action": "prepare_messages"})
        
        # Create a chat completion
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=together_messages
        )
        
        log["steps"].append({"action": "create_chat_completion", "response_id": response.id})
        
        return response.choices[0].message.content.strip(), log

class GroqModel(AIModel):
    def __init__(self, api_key, model_name="llama3-8b-8192"):
        super().__init__(model_name)
        self.client = Groq(api_key=api_key)
    
    def _generate_response(self, messages):
        log = {"steps": []}

        log["steps"].append({"action": "prepare_messages"})

        # Create a chat completion
        chat_completion = self.client.chat.completions.create(
            messages=messages,
            model=self.model_name
        )

        log["steps"].append({"action": "create_chat_completion"})

        return chat_completion.choices[0].message.content.strip(), log