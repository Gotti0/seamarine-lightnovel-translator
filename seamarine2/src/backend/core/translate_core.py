from google import genai
from google.genai import types
from logger_config import setup_logger
import logging
import os
from backend.model import AiModelConfig
import ast
import time
import re
import json

class TranslateCore:
    def __init__(self, language_from=""):
        self._logger = logging.getLogger("seamarine_translate")
        try:
            self.language_from = language_from
            self._keys: list[str] = []
            self._current_key_index: int = 0
            self._client = None
            self._model_data: AiModelConfig
            self._logger.info(str(self) + ".__init__")
        except Exception as e:
            self._logger.error(str(self) + str(e))
        
    def register_keys(self, keys: list[str]) -> bool: 
        try:
            self._keys = keys
            if not self._keys:
                return False
            
            os.environ['GOOGLE_API_KEY'] = self._keys[0]
            self._client = genai.Client()
            self._logger.info(str(self) + f".register_keys(len(keys)={len(keys)})")
            return True
        except Exception as e:
            self._logger.error(str(self) + f".register_keys(len(keys)={len(keys)})\n-> " + str(e))
            return False
        
    def update_model_data(self, data: AiModelConfig) -> bool:
        try:
            self._model_data = data
            self._logger.info(str(self) + f".update_model_data({str(data.to_dict())})")
            return True
        except Exception as e:
            self._logger.error(str(self) + f".update_model_data({str(data.to_dict())})\n-> " + str(e))
            return False

    def get_model_list(self) -> list[str]:
        try:
            raw_model_list = self._client.models.list()
            model_list = [
                (model.name).removeprefix("models/")
                for model in raw_model_list
                for action in model.supported_actions
                if action == 'generateContent'
            ]
            self._logger.info(str(self) + ".get_model_list -> " + str(model_list))
            return model_list
        except Exception as e:
            self._logger.error(str(self) + str(e))
            return []
        
    def rotate_to_next_key(self) -> bool:
        """
        다음 API 키로 로테이션합니다.
        Worker가 429 에러 처리 시 명시적으로 호출합니다.
        
        Returns:
            bool: 로테이션 성공 여부 (키가 1개 이하면 False)
        """
        if not self._keys or len(self._keys) <= 1:
            self._logger.warning("Cannot rotate: insufficient keys")
            return False
        
        self._current_key_index = (self._current_key_index + 1) % len(self._keys)
        new_key = self._keys[self._current_key_index]
        os.environ['GOOGLE_API_KEY'] = new_key
        self._client = genai.Client()
        self._logger.info(f"Rotated to key index {self._current_key_index}")
        return True
    
    def has_more_keys(self) -> bool:
        """로테이션 가능한 키가 있는지 확인"""
        return len(self._keys) > 1
    
    def get_keys_count(self) -> int:
        """등록된 전체 키 개수 반환"""
        return len(self._keys)
    
    def get_current_key_index(self) -> int:
        """현재 사용 중인 키의 인덱스 반환"""
        return self._current_key_index

    def _parse_retry_delay_from_error(self, error, extra_seconds=2) -> int:
        """
        API 에러 응답에서 retryDelay를 파싱합니다.
        """
        error_str = str(error)
        retry_seconds = 0
        
        pattern = r'["\']?retryDelay["\']?\s*:\s*["\']?(\d+(?:\.\d+)?)[sS]?["\']?'
        
        try:
            match = re.search(pattern, error_str)
            if match:
                val = float(match.group(1))
                retry_seconds = int(val)
                self._logger.info(f"Found retryDelay in error message: {retry_seconds}s")
        except Exception as e:
            self._logger.warning(f"retryDelay 정규식 추출 실패: {e}")
            return 0

        return retry_seconds + extra_seconds if retry_seconds > 0 else 0

    def _calculate_backoff_delay(self, backoff_attempt: int, error_str: str = "", base_delay: int = 10) -> int:
        """
        지수 백오프 딜레이를 계산합니다.
        """
        if error_str:
            api_delay = self._parse_retry_delay_from_error(error_str)
            if api_delay > 0:
                self._logger.info(f"Using API retryDelay: {api_delay}s")
                return api_delay
        
        delay = base_delay * (2 ** backoff_attempt)
        delay = min(delay, 300)
        
        self._logger.info(f"Using exponential backoff: {delay}s (attempt {backoff_attempt})")
        return delay

    def generate_content(self, contents: str | bytes, divide_n_conquer = True, resp_in_json = False, level: int = 0, schema: dict = None, retry_delay: int = 10):
        """
        Gemini API를 호출하여 콘텐츠를 생성합니다.
        내부적으로 재시도 및 키 로테이션을 수행합니다.
        """
        # API 호출 전 딜레이 적용 (재귀 호출 시에도 적용됨)
        if retry_delay > 0:
            time.sleep(retry_delay)

        total_keys = self.get_keys_count()
        max_keys_to_try = total_keys if total_keys > 0 else 1
        last_exception = None

        for key_attempt in range(max_keys_to_try):
            for attempt in range(3):
                try:
                    gen_config = types.GenerateContentConfig(
                        max_output_tokens= 65536 if '2.5' in self._model_data.name else 8192,
                        system_instruction= self._model_data.system_prompt,
                        temperature= self._model_data.temperature,
                        top_p= self._model_data.top_p,
                        frequency_penalty= self._model_data.frequency_penalty,
                        safety_settings = [
                            types.SafetySetting(
                                category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                                threshold=types.HarmBlockThreshold.OFF
                            ),
                            types.SafetySetting(
                                category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                                threshold=types.HarmBlockThreshold.OFF
                            ),
                            types.SafetySetting(
                                category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                                threshold=types.HarmBlockThreshold.OFF
                            ),
                            types.SafetySetting(
                                category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                                threshold=types.HarmBlockThreshold.OFF
                            ),
                            types.SafetySetting(
                                category=types.HarmCategory.HARM_CATEGORY_CIVIC_INTEGRITY,
                                threshold=types.HarmBlockThreshold.OFF
                            )
                        ],
                    )
                    if self._model_data.use_thinking_budget:
                        gen_config.thinking_config = types.ThinkingConfig(thinking_budget=self._model_data.thinking_budget)

                    if resp_in_json and schema:
                        gen_config.response_mime_type = "application/json"
                        gen_config.response_json_schema = schema
                    
                    resp = self._client.models.generate_content(
                        model=self._model_data.name,
                        contents=contents,
                        config=gen_config
                    )

                    if resp.prompt_feedback and resp.prompt_feedback.block_reason:
                        self._logger.warning(f"Response blocked with the reason {resp.prompt_feedback.block_reason}")
                        if divide_n_conquer and isinstance(contents, str):
                            return self._divide_and_conquer_json(contents, level=level, schema=schema, retry_delay=retry_delay) if resp_in_json else self._divide_and_conquer(contents, level=level, retry_delay=retry_delay)
                        else:
                            return ""
                    
                    return self._clean_gemini_response(resp.text)

                except Exception as e:
                    last_exception = e
                    error_str = str(e)
                    error_lower = error_str.lower()
                    self._logger.error(f"{str(self)}.generate_content -> {error_str}")

                    # 429, 503 등 재시도 필요한 에러
                    is_rate_limit = "429" in error_str or "resource exhausted" in error_lower or "quota" in error_lower
                    is_server_error = "503" in error_str or "unavailable" in error_lower
                    
                    if is_rate_limit or is_server_error:
                        if attempt < 2:
                            delay = self._calculate_backoff_delay(attempt, error_str, retry_delay)
                            time.sleep(delay)
                            continue
                        # 3회 실패 시 루프 종료하고 키 로테이션으로 넘어감

                    # 400, 500 에러는 divide and conquer로 처리 (단, 503 제외)
                    # 500 INTERNAL은 일시적일 수 있으나 기존 로직 유지하여 divide and conquer 시도
                    elif ("400" in error_str or "INVALID_ARGUMENT" in error_str or "500" in error_str or "INTERNAL" in error_str):
                        self._logger.warning(f"400/500 error detected. Attempting divide and conquer.")
                        if divide_n_conquer and isinstance(contents, str):
                            return self._divide_and_conquer_json(contents, level=level, schema=schema, retry_delay=retry_delay) if resp_in_json else self._divide_and_conquer(contents, level=level, retry_delay=retry_delay)
                        else:
                            return ""
                    
                    else:
                        # 기타 에러
                        if attempt < 2:
                            time.sleep(retry_delay)
                            continue
            
            # 현재 키에서 3회 실패함. 다음 키로 로테이션 시도
            if key_attempt < max_keys_to_try - 1:
                if self.rotate_to_next_key():
                    self._logger.info(f"Rotating to next key due to failure (Attempt {key_attempt + 1}/{max_keys_to_try})")
                    continue
                else:
                    self._logger.error("Failed to rotate key")
                    break
        
        # 모든 키 시도 실패
        raise last_exception
        
    def count_token(self, text):
        return self._client.models.count_tokens(text)
    

    def _divide_and_conquer_json(self, contents: str, level: int = 0, schema: dict = None, retry_delay: int = 10) -> str:
        self._logger.info(f"Start divide and conquer in json at level {level}")
        try:
            json_dict = json.loads(contents)
        except json.JSONDecodeError:
            self._logger.warning("Failed to parse JSON in divide_and_conquer_json. Falling back to plain text divide and conquer.")
            return self._divide_and_conquer(contents, level, retry_delay)

        keys = list(json_dict.keys())
        mid = len(keys) // 2
        first_half = {k: json_dict[k] for k in keys[:mid]}
        second_half = {k: json_dict[k] for k in keys[mid:]}

        self._logger.debug(f"First half: {first_half}")
        self._logger.debug(f"Second half: {second_half}")

        subcontents_1 = json.dumps(first_half)
        subcontents_2 = json.dumps(second_half)

        schema1, schema2 = None, None
        if schema and "properties" in schema:
            try:
                schema1 = {
                    "type": "object",
                    "properties": {k: schema["properties"][k] for k in first_half.keys()},
                    "required": [str(k) for k in first_half.keys()]
                }
                schema2 = {
                    "type": "object",
                    "properties": {k: schema["properties"][k] for k in second_half.keys()},
                    "required": [str(k) for k in second_half.keys()]
                }
            except KeyError as e:
                self._logger.warning(f"Failed to create sub-schema due to KeyError: {e}. Proceeding without schema.")
                schema1, schema2 = None, None


        subdict_1 = {}
        subdict_2 = {}
        
        try:
            self._logger.info(f"Processing first half... (level {level + 1})")
            subresp_1 = self.generate_content(subcontents_1, resp_in_json=True, level=level+1, schema=schema1, retry_delay=retry_delay)
            subdict_1.update(json.loads(subresp_1))
            self._logger.debug(f"Response from first half: {subresp_1}")
        except Exception as e:
            self._logger.error(f"Failed to process first half in divide_and_conquer: {e}")
            raise

        try:
            self._logger.info(f"Processing second half... (level {level + 1})")
            subresp_2 = self.generate_content(subcontents_2, resp_in_json=True, level=level+1, schema=schema2, retry_delay=retry_delay)
            subdict_2.update(json.loads(subresp_2))
            self._logger.debug(f"Response from second half: {subresp_2}")
        except Exception as e:
            self._logger.error(f"Failed to process second half in divide_and_conquer: {e}")
            raise

        merged_dict = subdict_1.copy()
        for k, v in subdict_2.items():
            if k in merged_dict and merged_dict[k] != v:
                self._logger.warning(f"Duplicate key '{k}' detected during merge in divide_and_conquer_json. Overwriting '{merged_dict[k]}' with '{v}'")
            merged_dict[k] = v
            
        self._logger.debug(f"Merged dict: {merged_dict}")

        self._logger.info(f"End divide and conquer in json at level {level}")
        return json.dumps(merged_dict)
    
    def _divide_and_conquer(self, contents: str, level: int = 0, retry_delay: int = 10) -> str:
        self._logger.info(f"Start divide and conquer at level {level}")
        subcontents_1 = contents[0:len(contents)//2]
        subcontents_2 = contents[len(contents)//2:len(contents)]

        try:
            resp1 = self.generate_content(subcontents_1, level=level+1, retry_delay=retry_delay)
        except Exception as e:
            self._logger.error(f"Failed to process first half in divide_and_conquer: {e}")
            raise

        try:
            resp2 = self.generate_content(subcontents_2, level=level+1, retry_delay=retry_delay)
        except Exception as e:
            self._logger.error(f"Failed to process second half in divide_and_conquer: {e}")
            raise

        # JSON 응답인 경우 병합 시도
        try:
            json1 = json.loads(resp1)
            json2 = json.loads(resp2)
            if isinstance(json1, dict) and isinstance(json2, dict):
                merged = json1.copy()
                for k, v in json2.items():
                    if k in merged and merged[k] != v:
                        self._logger.warning(f"Duplicate key '{k}' detected during merge in divide_and_conquer. Overwriting '{merged[k]}' with '{v}'")
                    merged[k] = v
                return json.dumps(merged, ensure_ascii=False)
        except json.JSONDecodeError:
            pass

        return resp1 + resp2


    def _clean_gemini_response(self, response_text: str) -> str:
        """
        Cleans the Gemini API response by stripping markdown formatting.
        """
        text = response_text.strip()
        if text.startswith("```html"):
            text = text[7:].strip()
        elif text.startswith("```json"):
            text = text[7:].strip()
        elif text.startswith("```"):
            text = text[3:].strip()
        if text.startswith("### html"):
            text = text[8:].strip()
        if text.endswith("```"):
            text = text[:-3].strip()

        self.escape_special_chars_in_json_string_safe(text)
            
        return text
    
    def escape_special_chars_in_json_string_safe(self, s: str) -> str:
        """
        JSON과 유사한 문자열에서 큰따옴표로 묶인 값 내부의 특수문자들을
        '안전하게' 이스케이프 처리합니다. 이미 이스케이프된 문자는 건드리지 않습니다.

        :param s: JSON 형태를 띤 입력 문자열
        :return: 특수문자가 올바르게 이스케이프 처리된 문자열
        """
        
        # 이스케이프 처리가 필요한 문자와 그 결과 매핑
        # 백슬래시를 맨 앞에 두는 것이 가독성과 논리에 좋습니다.
        escape_map = {
            '\\': '\\\\',
            '"': '\\"',
            '\n': '\\n',
            '\r': '\\r',
            '\t': '\\t',
            '\b': '\\b',
            '\f': '\\f',
        }

        def escape_match(match: re.Match) -> str:
            """re.sub에 사용될 바깥쪽 콜백 함수"""
            # "value" -> value
            content = match.group(1)

            def inner_replacer(m: re.Match) -> str:
                """문자열 내용물에 대해 실행될 안쪽 치환 함수"""
                # 그룹 1: 유효한 이스케이프 시퀀스 (예: \", \\, \n)
                if m.group(1):
                    # 이미 올바르므로 그대로 반환
                    return m.group(1)
                # 그룹 2: 이스케이프가 필요한 문자 (예: ", \, 개행문자)
                else:
                    char_to_escape = m.group(2)
                    return escape_map.get(char_to_escape, char_to_escape)

            # 안쪽 정규표현식:
            # 그룹 1: (\\.) -> 백슬래시(\)로 시작하는 모든 두 글자 문자(유효/무효 이스케이프 모두 포함)
            # 그룹 2: (["\\\n\r\t\b\f]) -> 이스케이프가 필요한 문자들의 집합
            # | (OR) 로 두 그룹을 연결하여 둘 중 하나를 찾습니다.
            # 이 패턴은 `\`가 앞에 붙은 문자를 우선적으로 그룹 1로 매치하므로,
            # `"`는 `\"`의 일부가 아닐 때만 그룹 2로 매치됩니다.
            inner_pattern = r'(\\.)|(["\\\n\r\t\b\f])'
            
            # 문자열 내용물(content)에 대해서만 추가적인 치환 작업 수행
            escaped_content = re.sub(inner_pattern, inner_replacer, content)

            # 다시 큰따옴표로 감싸서 반환
            return f'"{escaped_content}"'

        # 바깥쪽 정규표현식: 큰따옴표로 감싸인 모든 부분을 찾음
        outer_pattern = r'"((?:[^"\\]|\\.)*)"'
        
        # re.sub를 사용하여 패턴에 맞는 모든 부분을 escape_match 함수의 결과로 치환
        return re.sub(outer_pattern, escape_match, s)