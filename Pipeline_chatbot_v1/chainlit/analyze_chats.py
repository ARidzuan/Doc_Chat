"""
Chat Log Analysis Script
Analyze chat logs to evaluate model/system performance
"""
import sys
from chat_logger import ChatLogAnalyzer

def main():
    analyzer = ChatLogAnalyzer()

    print("\n" + "="*70)
    print("CHAT LOG ANALYSIS")
    print("="*70)

    # Generate comprehensive report
    report = analyzer.generate_report("./data/chat_analysis_report.json")

    if "error" in report:
        print(f"\n{report['error']}")
        return

    # Display summary
    summary = report['summary']
    print(f"\n📊 OVERALL STATISTICS")
    print(f"{'─'*70}")
    print(f"Total Chat Sessions:        {summary['total_sessions']}")
    print(f"Total Messages:             {summary['total_messages']}")
    print(f"  - User Messages:          {summary['total_user_messages']}")
    print(f"  - Assistant Messages:     {summary['total_assistant_messages']}")
    print(f"Avg Messages per Session:   {summary['avg_messages_per_session']:.1f}")

    if summary['avg_response_time_seconds']:
        print(f"Avg Response Time:          {summary['avg_response_time_seconds']:.2f} seconds")

    # Display individual sessions
    print(f"\n💬 SESSION DETAILS")
    print(f"{'─'*70}")

    for i, session in enumerate(report['sessions'], 1):
        print(f"\n[{i}] {session['file']}")
        print(f"    Session ID: {session['session_id'][:16]}...")
        print(f"    Start Time: {session['start_time']}")

        if session['duration_seconds']:
            minutes = session['duration_seconds'] / 60
            print(f"    Duration:   {minutes:.1f} minutes")

        print(f"    Messages:   {session['user_messages']} user, {session['assistant_messages']} assistant")

        if session['avg_response_time']:
            print(f"    Avg Response: {session['avg_response_time']:.2f}s "
                  f"(min: {session['min_response_time']:.2f}s, max: {session['max_response_time']:.2f}s)")

        if session['errors'] > 0:
            print(f"    ⚠️  Errors: {session['errors']}")

        # Show metadata if available
        if session.get('metadata'):
            meta = session['metadata']
            if 'model' in meta:
                print(f"    Model:      {meta['model']}")
            if 'mode' in meta:
                print(f"    Mode:       {meta['mode']}")

    print(f"\n{'='*70}")
    print(f"✅ Full report saved to: ./data/chat_analysis_report.json")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
