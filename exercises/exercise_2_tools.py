"""Bài Tập 2: Thêm Tools và Knowledge Base

Hoàn thành các TODO để thêm tool và knowledge base entry mới.
"""

import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

from common.llm import get_llm

# Knowledge base
LEGAL_KNOWLEDGE = [
    {
        "id": "ucc_breach",
        "keywords": ["breach", "contract", "remedies", "damages", "ucc"],
        "text": (
            "Under the Uniform Commercial Code (UCC) Article 2, remedies for breach of contract "
            "include: (1) expectation damages — placing the non-breaching party in the position "
            "they would have been in had the contract been performed; (2) consequential damages "
            "for foreseeable losses (Hadley v. Baxendale, 1854); (3) specific performance when "
            "the subject matter is unique; (4) cover damages — the cost of obtaining substitute "
            "performance. The statute of limitations is typically 4 years (UCC § 2-725)."
        ),
    },
    {
        "id": "nda_trade_secret",
        "keywords": ["nda", "non-disclosure", "confidential", "trade secret", "agreement"],
        "text": (
            "NDA breaches may trigger both contractual and statutory liability. Under the Defend "
            "Trade Secrets Act (DTSA, 18 U.S.C. § 1836), misappropriation of trade secrets can "
            "result in: (1) injunctive relief; (2) actual damages plus unjust enrichment; "
            "(3) exemplary damages up to 2x actual damages for willful misappropriation; "
            "(4) attorney's fees. State Uniform Trade Secrets Act (UTSA) versions provide "
            "additional remedies. Criminal prosecution is possible under the Economic Espionage "
            "Act (18 U.S.C. § 1832) with penalties up to $5M for individuals."
        ),
    },
    {
        "id": "dtsa_details",
        "keywords": ["dtsa", "federal", "trade secret", "defend", "statute"],
        "text": (
            "The Defend Trade Secrets Act (2016) created a federal private cause of action for "
            "trade secret misappropriation. Key provisions: (1) ex parte seizure orders in "
            "extraordinary circumstances; (2) 3-year statute of limitations; (3) immunity for "
            "whistleblower disclosures to government officials; (4) employers must notify "
            "employees of whistleblower immunity in any NDA or employment agreement."
        ),
    },
    {
        "id": "liquidated_damages",
        "keywords": ["liquidated", "damages", "penalty", "clause", "contract", "nda"],
        "text": (
            "Liquidated damages clauses in NDAs are enforceable if: (1) actual damages would be "
            "difficult to calculate at the time of contracting; (2) the stipulated amount is a "
            "reasonable estimate of anticipated harm. Courts will void clauses that function as "
            "penalties (Restatement (Second) of Contracts § 356). Typical NDA liquidated damages "
            "range from $10,000 to $500,000 depending on the nature of the confidential information."
        ),
    },
    {
        "id": "injunctive_relief",
        "keywords": ["injunction", "restraining", "order", "equitable", "nda", "breach"],
        "text": (
            "Courts routinely grant temporary restraining orders (TROs) and preliminary injunctions "
            "for NDA breaches because: (1) confidential information, once disclosed, cannot be "
            "'un-disclosed' — making monetary damages inadequate; (2) irreparable harm is presumed "
            "for trade secret misappropriation in many jurisdictions. The movant must show "
            "likelihood of success on the merits, irreparable harm, balance of equities, and "
            "public interest (Winter v. Natural Resources Defense Council, 2008)."
        ),
    },
    {
    "id": "labor_law",
    "keywords": ["lao động", "sa thải", "hợp đồng lao động", "labor", "termination"],
    "text": (
        "Theo Bộ luật Lao động Việt Nam 2019, người sử dụng lao động có thể "
        "đơn phương chấm dứt hợp đồng trong các trường hợp: (1) người lao động "
        "thường xuyên không hoàn thành công việc; (2) bị ốm đau, tai nạn đã điều trị "
        "12 tháng chưa khỏi; (3) thiên tai, hỏa hoạn; (4) người lao động đủ tuổi nghỉ hưu."
    ),
}
]


@tool
def search_legal_knowledge(query: str) -> str:
    """Tìm kiếm trong knowledge base pháp lý."""
    query_lower = query.lower()
    for entry in LEGAL_KNOWLEDGE:
        if any(kw in query_lower for kw in entry["keywords"]):
            return f"[{entry['id']}] {entry['text']}"
    return "Không tìm thấy thông tin liên quan."


@tool
def check_statute_of_limitations(case_type: str) -> str:
    """Kiểm tra thời hiệu khởi kiện theo loại vụ án.
    
    Args:
        case_type: Loại vụ án (contract, tort, property)
    """
    limits = {
        "contract": "4 năm (UCC § 2-725)",
        "tort": "2-3 năm tùy bang",
        "property": "5 năm",
    }
    return limits.get(case_type.lower(), "Không xác định")


async def main():
    load_dotenv()
    llm = get_llm()
    
    tools = [search_legal_knowledge, check_statute_of_limitations]
    llm_with_tools = llm.bind_tools(tools)
    
    question = "Thời hiệu khởi kiện vụ vi phạm hợp đồng là bao lâu?"
    
    messages = [
        SystemMessage(content="Bạn là chuyên gia pháp lý. Sử dụng tools để tra cứu thông tin."),
        HumanMessage(content=question),
    ]
    
    print(f"Câu hỏi: {question}\n")
    
    # First LLM call - decide which tools to use
    response = await llm_with_tools.ainvoke(messages)
    messages.append(response)
    
    # Execute tools if requested
    if response.tool_calls:
        for tool_call in response.tool_calls:
            print(f"🔧 Gọi tool: {tool_call['name']}")
            tool_result = None
            
            if tool_call["name"] == "search_legal_knowledge":
                tool_result = search_legal_knowledge.invoke(tool_call["args"])
            elif tool_call["name"] == "check_statute_of_limitations":
                tool_result = check_statute_of_limitations.invoke(tool_call["args"])
            
            if tool_result:
                messages.append(ToolMessage(content=tool_result, tool_call_id=tool_call["id"]))
        
        # Second LLM call - synthesize final answer
        final_response = await llm_with_tools.ainvoke(messages)
        print(f"\n✅ Kết quả:\n{final_response.content}")
    else:
        print(f"\n✅ Kết quả:\n{response.content}")


if __name__ == "__main__":
    asyncio.run(main())
