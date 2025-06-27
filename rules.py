# rules.py
from durable.lang import *

rule_results = {}

with ruleset('fraud'):

    @when_all(m.Amount > 100)
    def high_amount(c):
        result = {'rule': 'high_amount', 'risk': 'medium'}
        rule_results[c.m['__id__']] = result
        c.s.result = result
        print("🔥 [规则命中] high_amount")

    @when_all(m.V14 > 1.5)
    def suspicious_v14_high(c):
        result = {'rule': 'v14_unusual_high', 'risk': 'medium'}
        rule_results[c.m['__id__']] = result
        c.s.result = result
        print("🔥 [规则命中] suspicious_v14_high")

    @when_all((m.V17 > 0.5) & (m.V10 < -0.5))
    def abnormal_combination(c):
        result = {'rule': 'v17up_v10down', 'risk': 'high'}
        rule_results[c.m['__id__']] = result
        c.s.result = result
        print("🔥 [规则命中] abnormal_combination")

    @when_all((m.Amount < 5) & (m.V12 < -2))
    def small_amount_with_signal(c):
        result = {'rule': 'small_amt_signal', 'risk': 'medium'}
        rule_results[c.m['__id__']] = result
        c.s.result = result
        print("🔥 [规则命中] small_amount_with_signal")

    # -------------------------
    # 🌲 决策树规则翻译如下：
    # -------------------------

    @when_all((m.V14 <= -1.81) & (m.V4 <= -0.48) & (m.V14 <= -1.83))
    def rule_tree_1(c):
        result = {'rule': 'tree_rule_1', 'risk': 'low'}
        rule_results[c.m['__id__']] = result
        print("🌲 [规则命中] tree_rule_1")

    @when_all((m.V14 <= -1.81) & (m.V4 <= -0.48) & (m.V14 > -1.83) & (m.V14 <= -1.82))
    def rule_tree_2(c):
        result = {'rule': 'tree_rule_2', 'risk': 'high'}
        rule_results[c.m['__id__']] = result
        print("🌲 [规则命中] tree_rule_2")

    @when_all((m.V14 <= -1.81) & (m.V4 <= -0.48) & (m.V14 > -1.82))
    def rule_tree_3(c):
        result = {'rule': 'tree_rule_3', 'risk': 'low'}
        rule_results[c.m['__id__']] = result
        print("🌲 [规则命中] tree_rule_3")

    @when_all((m.V14 <= -1.81) & (m.V4 > -0.48) & (m.V1 <= 1.99) & (m.V10 <= 0.57))
    def rule_tree_4(c):
        result = {'rule': 'tree_rule_4', 'risk': 'high'}
        rule_results[c.m['__id__']] = result
        print("🌲 [规则命中] tree_rule_4")

    @when_all((m.V14 <= -1.81) & (m.V4 > -0.48) & (m.V1 <= 1.99) & (m.V10 > 0.57))
    def rule_tree_5(c):
        result = {'rule': 'tree_rule_5', 'risk': 'low'}
        rule_results[c.m['__id__']] = result
        print("🌲 [规则命中] tree_rule_5")

    @when_all((m.V14 <= -1.81) & (m.V4 > -0.48) & (m.V1 > 1.99))
    def rule_tree_6(c):
        result = {'rule': 'tree_rule_6', 'risk': 'low'}
        rule_results[c.m['__id__']] = result
        print("🌲 [规则命中] tree_rule_6")

    @when_all((m.V14 > -1.81) & (m.V4 <= 1.66) & (m.V20 <= -1.03) & (m.V12 <= -0.08))
    def rule_tree_7(c):
        result = {'rule': 'tree_rule_7', 'risk': 'high'}
        rule_results[c.m['__id__']] = result
        print("🌲 [规则命中] tree_rule_7")

    @when_all((m.V14 > -1.81) & (m.V4 <= 1.66) & (m.V20 <= -1.03) & (m.V12 > -0.08))
    def rule_tree_8(c):
        result = {'rule': 'tree_rule_8', 'risk': 'low'}
        rule_results[c.m['__id__']] = result
        print("🌲 [规则命中] tree_rule_8")

    @when_all((m.V14 > -1.81) & (m.V4 <= 1.66) & (m.V20 > -1.03) & (m.V7 <= 2.88))
    def rule_tree_9(c):
        result = {'rule': 'tree_rule_9', 'risk': 'low'}
        rule_results[c.m['__id__']] = result
        print("🌲 [规则命中] tree_rule_9")

    @when_all((m.V14 > -1.81) & (m.V4 <= 1.66) & (m.V20 > -1.03) & (m.V7 > 2.88))
    def rule_tree_10(c):
        result = {'rule': 'tree_rule_10', 'risk': 'high'}
        rule_results[c.m['__id__']] = result
        print("🌲 [规则命中] tree_rule_10")

    @when_all((m.V14 > -1.81) & (m.V4 > 1.66) & (m.V8 <= -0.22) & (m.V13 <= 0.37))
    def rule_tree_11(c):
        result = {'rule': 'tree_rule_11', 'risk': 'high'}
        rule_results[c.m['__id__']] = result
        print("🌲 [规则命中] tree_rule_11")

    @when_all((m.V14 > -1.81) & (m.V4 > 1.66) & (m.V8 <= -0.22) & (m.V13 > 0.37))
    def rule_tree_12(c):
        result = {'rule': 'tree_rule_12', 'risk': 'low'}
        rule_results[c.m['__id__']] = result
        print("🌲 [规则命中] tree_rule_12")

    @when_all((m.V14 > -1.81) & (m.V4 > 1.66) & (m.V8 > -0.22) & (m.V6 <= 0.10))
    def rule_tree_13(c):
        result = {'rule': 'tree_rule_13', 'risk': 'high'}
        rule_results[c.m['__id__']] = result
        print("🌲 [规则命中] tree_rule_13")

    @when_all((m.V14 > -1.81) & (m.V4 > 1.66) & (m.V8 > -0.22) & (m.V6 > 0.10))
    def rule_tree_14(c):
        result = {'rule': 'tree_rule_14', 'risk': 'low'}
        rule_results[c.m['__id__']] = result
        print("🌲 [规则命中] tree_rule_14")

def get_rule_result(msg_id):
    return rule_results.pop(msg_id, None)

print("[rules.py] 规则系统已加载 ✔")

def start_rules():
    print("[rules.py] 规则引擎已启动 ✔")

