"""
测试自愈流程脚本
输入: docs/脚本失败日志示例.md
输出: tickets/ 目录下的分析报告
"""

import sys
from pathlib import Path

# 将项目根目录添加到 sys.path，确保无论从哪个目录运行都能找到模块
_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import re

from self_healing.fast_path import FastPathHealer
from self_healing.slow_path import SlowPathHealer
from core.llm import create_llm


def parse_failure_log(log_path: str) -> list:
    """
    解析失败日志，提取每个失败用例的 log_info

    Args:
        log_path: 失败日志文件路径

    Returns:
        List[dict]: 每个失败用例的 log_info 列表
    """
    content = Path(log_path).read_text(encoding="utf-8")

    # 分割每个失败用例
    # 匹配 "❌ 失败用例 N：..." 或 "## ❌ 失败用例 N：..."
    pattern = r'## ❌ 失败用例 \d+：[^\n]+\n(.*?)(?=## ❌ 失败用例|\Z)'
    sections = re.split(r'(?=## ❌ 失败用例)', content)

    log_infos = []

    for section in sections:
        if not section.strip() or '失败用例' not in section:
            continue

        # 提取用例ID（Tc_Func_Nodepool_001 格式）
        case_id_match = re.search(r'失败用例 \d+：(\S+)', section)
        case_id = case_id_match.group(1).strip() if case_id_match else "未知"

        # 提取 Scenario 名称 - 从 Gherkin 的 Scenario: 行
        scenario_match = re.search(r'^\s*Scenario:\s*(.+)\s*$', section, re.MULTILINE)
        scenario = scenario_match.group(1).strip() if scenario_match else case_id

        # 提取所属模块 - 格式为：**所属模块**：`Node`
        module_match = re.search(r'\*\*所属模块\*\*.*?`([^`]+)`', section)
        module = module_match.group(1).strip() if module_match else "Unknown"

        # 提取失败步骤 - 格式为：**失败步骤 (Failed Step)**：`When xxx`
        failed_step_match = re.search(r'\*\*失败步骤.*?`([^`]+)`', section)
        failed_step = failed_step_match.group(1).strip() if failed_step_match else ""

        # 提取报错信息 - 从 E   开头的行（Python traceback格式）
        error_lines = re.findall(r'^E\s+.+$', section, re.MULTILINE)
        error_msg = '\n'.join(error_lines[:8]) if error_lines else ""

        # 如果 E   行太少或关键信息不全，尝试提取 JSON body 中的 message 字段
        if len(error_lines) < 5 or 'is invalid' not in error_msg:
            msg_in_json = re.search(r'"message":\s*"([^"]{20,})"', section)
            if msg_in_json:
                error_msg = msg_in_json.group(1).replace('\\n', ' ')[:500]

        # 如果还是没有，尝试从 ```text 块提取
        if not error_msg:
            text_block_match = re.search(r'```text\s*\n(.*?)\n```', section, re.DOTALL)
            if text_block_match:
                error_msg = text_block_match.group(1).strip()[:500]

        # 提取错误类型（根据错误特征）
        error_type = "error"
        msg_for_classify = error_msg.lower()
        if 'KeyError'.lower() in msg_for_classify:
            error_type = "error"
        elif 'Quota' in msg_for_classify or 'exceeded' in msg_for_classify:
            error_type = "error"
        elif 'deprecated' in msg_for_classify or 'removed' in msg_for_classify or 'field is immutable' in msg_for_classify:
            error_type = "error"
        elif 'cgroup' in msg_for_classify or 'containerd' in msg_for_classify:
            error_type = "error"

        log_info = {
            'case_id': case_id,
            'scenario': scenario,
            'module': module,
            'step_name': failed_step,
            'type': error_type,
            'message': error_msg[:800],
            'raw_section': section[:2000]
        }

        log_infos.append(log_info)
        print(f"  解析到失败用例: {case_id} | {scenario} (模块: {module})")

    return log_infos


def main():
    """主流程"""
    print("=" * 60)
    print("BDD-RAG-System 自愈流程测试")
    print("=" * 60)

    # 读取失败日志 - 使用脚本所在目录作为基准
    project_root = Path(__file__).parent.parent
    failure_log_path = project_root / "docs" / "脚本失败日志示例.md"
    if not failure_log_path.exists():
        print(f"❌ 失败日志文件不存在: {failure_log_path}")
        return

    print(f"\n📖 读取失败日志: {failure_log_path}")
    log_infos = parse_failure_log(failure_log_path)
    print(f"  共解析到 {len(log_infos)} 个失败用例")

    if not log_infos:
        print("❌ 未解析到失败用例")
        return

    # 初始化
    print("\n🔧 初始化组件...")
    llm = create_llm()
    fast_healer = FastPathHealer(llm=llm)

    kb3_dir = project_root / "knowledge-base/kb3_bug_reports/indexed/chroma_db"
    if kb3_dir.exists():
        slow_healer = SlowPathHealer(llm=llm, kb3_dir=str(kb3_dir))
        print(f"  ✅ SlowPathHealer 初始化完成 (KB3: {kb3_dir})")
    else:
        print(f"  ⚠️ KB3 目录不存在: {kb3_dir}，Slow Path 将只使用 LLM 分析")
        slow_healer = SlowPathHealer(llm=llm)

    # 确保 tickets 目录存在
    tickets_dir = project_root / "tickets"
    tickets_dir.mkdir(exist_ok=True)

    # 处理每个失败用例
    results = []
    for i, log_info in enumerate(log_infos, 1):
        print(f"\n{'=' * 60}")
        print(f"处理失败用例 {i}/{len(log_infos)}: {log_info.get('scenario')}")
        print(f"{'=' * 60}")

        # Fast Path 诊断
        print("\n🔍 Fast Path 诊断...")
        fixed, msg, ticket_path, category = fast_healer.diagnose_and_heal(log_info)

        if fixed:
            print(f"  ✅ Fast Path 直接判定: {msg}")
            if ticket_path:
                print(f"     报告: {ticket_path}")
            results.append({
                'case_id': log_info.get('case_id', ''),
                'scenario': log_info.get('scenario'),
                'path': 'Fast Path',
                'status': 'fixed',
                'message': msg,
                'ticket_path': ticket_path
            })
            continue

        print(f"  ⚠️ Fast Path 无法修复: {msg}")
        print(f"     Fast Path 分类: {category}")

        # Slow Path 深度分析 - 传入 Fast Path 分类结果和 case_id
        print("\n🔍 Slow Path 深度分析...")
        _, report_msg, ticket_path = slow_healer.diagnose_and_heal(
            log_info,
            fast_path_category=category,
            case_id=log_info.get('case_id', '')
        )

        if ticket_path:
            print(f"  ✅ 生成分析报告: {ticket_path}")
            results.append({
                'case_id': log_info.get('case_id', ''),
                'scenario': log_info.get('scenario'),
                'path': 'Slow Path',
                'status': 'reported',
                'message': report_msg,
                'ticket_path': ticket_path
            })
        else:
            print(f"  ⚠️ Slow Path 分析失败: {report_msg}")
            results.append({
                'case_id': log_info.get('case_id', ''),
                'scenario': log_info.get('scenario'),
                'path': 'Slow Path',
                'status': 'failed',
                'message': report_msg
            })

    # 汇总报告
    print("\n" + "=" * 60)
    print("📊 处理结果汇总")
    print("=" * 60)
    for r in results:
        status_icon = "✅" if r['status'] == 'fixed' else "📋"
        case_id = r.get('case_id', r['scenario'])
        print(f"  {status_icon} [{case_id}] {r['scenario']}")
        print(f"      路径: {r['path']} | 状态: {r['status']}")
        print(f"      消息: {r['message'][:80]}...")
        if 'ticket_path' in r and r['ticket_path']:
            print(f"      报告: {r['ticket_path']}")

    print(f"\n📁 分析报告目录: {tickets_dir.absolute()}")

    # 统计各类报告数量
    categories = {
        'script_error': '脚本问题',
        'env_fault': '环境故障',
        'resource_insufficient': '资源不足',
        'version_bug': '版本Bug'
    }
    for sub_dir, desc in categories.items():
        files = list((tickets_dir / sub_dir).glob("*.md"))
        if files:
            print(f"  {desc}: {len(files)} 份")

    ticket_files = list(tickets_dir.glob("**/*.md"))
    print(f"  共生成 {len(ticket_files)} 份分析报告")


if __name__ == "__main__":
    main()
