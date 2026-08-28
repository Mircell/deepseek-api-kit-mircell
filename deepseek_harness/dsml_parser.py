import re
import json
import uuid
from typing import List, Dict, Any, Optional

# الگوهای Regex برای تشخیص تگ‌های DSML و XML ساده
PATTERNS = {
    # تگ‌های ساده XML مانند <write>...</write> یا <read path="..."/>
    "simple_tag": r'<([a-zA-Z_][a-zA-Z0-9_-]*)>(.*?)</\1>',
    # تگ‌های DSML با کاراکترهای خاص
    "dsml_invoke": r'<｜DSML｜invoke\s+name="([^"]+)"\s*>(.*?)</｜DSML｜invoke>',
    "dsml_param": r'<｜DSML｜parameter\s+name="([^"]+)"\s+string="([^"]+)"\s*>(.*?)</｜DSML｜parameter>',
}

def parse_tool_calls_from_text(text: str) -> List[Dict[str, Any]]:
    """
    تشخیص تگ‌های ابزار (DSML یا XML ساده) و تبدیل به فرمت tool_calls استاندارد OpenAI.
    """
    tool_calls = []
    
    # 1. ابتدا تگ‌های DSML را بررسی کن
    for invoke_match in re.finditer(PATTERNS["dsml_invoke"], text, re.DOTALL):
        tool_name = invoke_match.group(1)
        invoke_body = invoke_match.group(2)
        
        # استخراج پارامترها
        arguments = {}
        for param_match in re.finditer(PATTERNS["dsml_param"], invoke_body, re.DOTALL):
            param_name = param_match.group(1)
            param_value = param_match.group(3).strip()
            arguments[param_name] = param_value
        
        # اگر پارامتری یافت نشد، سعی کن محتوا را JSON parse کن
        if not arguments:
            try:
                arguments = json.loads(invoke_body.strip())
            except:
                arguments = {"content": invoke_body.strip()}
        
        tool_calls.append({
            "id": f"toolu_{uuid.uuid4().hex[:8]}",
            "type": "function",
            "function": {
                "name": tool_name,
                "arguments": json.dumps(arguments, ensure_ascii=False)
            }
        })
    
    # 2. اگر تگ DSML یافت نشد، تگ‌های ساده XML را بررسی کن
    if not tool_calls:
        for match in re.finditer(PATTERNS["simple_tag"], text, re.DOTALL):
            tag_name = match.group(1)
            content = match.group(2).strip()
            
            # تلاش برای parse JSON
            try:
                arguments = json.loads(content)
            except:
                # اگر JSON نبود، کل محتوا را به عنوان یک پارامتر در نظر بگیر
                arguments = {"content": content}
            
            tool_calls.append({
                "id": f"toolu_{uuid.uuid4().hex[:8]}",
                "type": "function",
                "function": {
                    "name": tag_name,
                    "arguments": json.dumps(arguments, ensure_ascii=False)
                }
            })
    
    return tool_calls

def remove_tool_tags(text: str) -> str:
    """
    حذف کامل تمام تگ‌های ابزار (DSML و XML ساده) از متن.
    """
    # حذف تگ‌های DSML
    text = re.sub(PATTERNS["dsml_invoke"], '', text, flags=re.DOTALL)
    text = re.sub(r'<｜DSML｜[^>]+>', '', text)
    
    # حذف تگ‌های ساده XML
    text = re.sub(PATTERNS["simple_tag"], '', text, flags=re.DOTALL)
    
    # پاکسازی فاصله‌های اضافی
    return '\n'.join(line.strip() for line in text.splitlines() if line.strip())
