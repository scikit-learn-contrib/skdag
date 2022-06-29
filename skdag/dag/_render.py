import networkx as nx
import stackeddag.core as sd
from skdag.dag._utils import _is_passthrough

__all__ = ["DAGRenderer"]


class DAGRenderer:
    _EMPTY = "[empty]"
    _STYLES = {
        "light": {
            "node__color": "black",
            "node__fontcolor": "black",
            "edge__color": "black",
            "graph__bgcolor": "white",
        },
        "dark": {
            "node__color": "white",
            "node__fontcolor": "white",
            "edge__color": "white",
            "graph__bgcolor": "none",
        },
    }

    def __init__(self, dag, style=None):
        self.dag = dag
        if style is not None and style not in self._STYLES:
            raise ValueError(f"Unknown style: '{style}'.")
        self.style = style
        self.agraph = self.to_agraph()

    def _get_node_shape(self, estimator):
        if _is_passthrough(estimator):
            return "hexagon"
        if any(hasattr(estimator, attr) for attr in ["fit_predict", "predict"]):
            return "ellipse"
        return "box"

    def _is_empty(self):
        return len(self.dag) == 0

    def to_agraph(self):
        G = self.dag
        if self._is_empty():
            G = nx.DiGraph()
            G.add_node(
                self._EMPTY,
                shape="box",
            )

        A = nx.nx_agraph.to_agraph(G)
        A.graph_attr["rankdir"] = "LR"
        if self.style is not None:
            A.graph_attr.update(
                {
                    key.replace("graph__", ""): val
                    for key, val in self._STYLES[self.style].items()
                    if key.startswith("graph__")
                }
            )

        for v in G.nodes:
            anode = A.get_node(v)
            gnode = G.nodes[v]
            anode.attr["fontname"] = gnode.get("fontname", "sans")
            if self.style is not None:
                anode.attr.update(
                    {
                        key.replace("node__", ""): val
                        for key, val in self._STYLES[self.style].items()
                        if key.startswith("node__")
                    }
                )
            if "step" in gnode:
                anode.attr["shape"] = gnode.get(
                    "shape", self._get_node_shape(gnode["step"].estimator)
                )
                if gnode["step"].is_fitted:
                    anode.attr["peripheries"] = 2

        for u, v in G.edges:
            aedge = A.get_edge(u, v)
            if self.style is not None:
                aedge.attr.update(
                    {
                        key.replace("edge__", ""): val
                        for key, val in self._STYLES[self.style].items()
                        if key.startswith("edge__")
                    }
                )

        A.layout()
        return A

    def draw(self, format="svg", layout="dot"):
        A = self.agraph

        if format == "txt":
            if self._is_empty():
                return self._EMPTY

            # Edge case: single node, no edges.
            if len(A.edges()) == 0:
                return f"o    {next(A.nodes_iter())}\n"

            la = []
            for node in A.nodes():
                la.append((node, node))

            ed = []
            for src, dst in A.edges():
                ed.append((src, [dst]))

            return sd.edgesToText(sd.mkLabels(la), sd.mkEdges(ed))

        data = A.draw(format=format, prog=layout)

        if format == "svg":
            return data.decode(self.agraph.encoding)
        return data
