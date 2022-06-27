import networkx as nx
import stackeddag.core as sd
from sklearn.exceptions import NotFittedError
from sklearn.utils.estimator_checks import check_is_fitted

__all__ = ["DAGRenderer"]


class DAGRenderer:
    _EMPTY = "[empty]"
    COLORS = {
        "light_red": "#ff7fa7",
        "light_blue": "#80c4fc",
        "light_green": "#75e6a3",
        "light_grey": "#e6e6e6",
        "dark_grey": "#888888",
        "white": "#ffffff",
        "black": "#000000",
    }

    def __init__(self, dag):
        self.dag = dag
        self.agraph = self.to_agraph()

    def _get_node_color(self, G, node):
        if len(list(G.predecessors(node))) == 0:
            return self.COLORS["light_red"]
        if len(list(G.successors(node))) > 0:
            return self.COLORS["light_blue"]
        return self.COLORS["light_green"]

    def _get_node_shape(self, G, node):
        if len(list(G.predecessors(node))) == 0:
            return "ellipse"
        if len(list(G.successors(node))) > 0:
            return "box"
        return "ellipse"

    def _is_empty(self):
        return len(self.dag) == 0

    def to_agraph(self):
        G = self.dag
        if self._is_empty():
            G = nx.DiGraph()
            G.add_node(
                self._EMPTY,
                color=self.COLORS["light_grey"],
                fontcolor="black",
                shape="box",
            )

        A = nx.nx_agraph.to_agraph(G)
        A.graph_attr["rankdir"] = "LR"
        for v in G.nodes:
            anode = A.get_node(v)
            gnode = G.nodes[v]
            anode.attr["fontname"] = gnode.get("fontname", "sans")
            anode.attr["shape"] = gnode.get("shape", self._get_node_shape(G, v))
            anode.attr["style"] = gnode.get("style", "filled")
            color = self._get_node_color(G, v)
            anode.attr["bgcolor"] = gnode.get("color", color)
            anode.attr["color"] = gnode.get("color", color)
            anode.attr["fontcolor"] = gnode.get("fontcolor", self.COLORS["white"])
            if "step" in gnode:
                try:
                    check_is_fitted(gnode["step"].estimator)
                    anode.attr["label"] = f"{v}*"
                except NotFittedError:
                    pass

        for u, v in A.edges():
            aedge = A.get_edge(u, v)
            gedge = G.edges[u, v]
            aedge.attr["color"] = gedge.get("color", self.COLORS["dark_grey"])

        A.layout()
        return A

    def draw(self, format="svg", prog="dot"):
        A = self.agraph

        if format == "ascii":
            if self._is_empty():
                return self._EMPTY

            la = []
            for node in A.nodes():
                la.append((node, node))

            ed = []
            for src, dst in A.edges():
                ed.append((src, [dst]))

            return sd.edgesToText(sd.mkLabels(la), sd.mkEdges(ed))

        data = A.draw(format=format, prog="dot")

        if format == "svg":
            return data.decode(self.agraph.encoding)
        return data
