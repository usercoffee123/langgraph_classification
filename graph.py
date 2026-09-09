"""LangGraph orchestrates local detection, calculation, and Claude explanation."""

from langgraph.graph import END, START, StateGraph
from steps import State, calculate_occupancy, describe_detections, detect_objects, explain


def build_graph():
    builder = StateGraph(State)
    builder.add_node('detect', detect_objects)
    builder.add_node('describe_detections', describe_detections)
    builder.add_node('calculate', calculate_occupancy)
    builder.add_node('explain', explain)
    builder.add_edge(START, 'detect')
    builder.add_edge('detect', 'describe_detections')
    builder.add_edge('describe_detections', 'calculate')
    builder.add_edge('calculate', 'explain')
    builder.add_edge('explain', END)
    return builder.compile()


graph = build_graph()


def run(state: State) -> State:
    return graph.invoke(state)
