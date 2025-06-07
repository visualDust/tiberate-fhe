import inspect
from functools import wraps
from time import time

import pandas as pd
import plotly.express as px
from loguru import logger

from tiberate import CkksEngine
from tiberate.typing import *  # noqa: F403


class CkksEngineEventTracer:
    @property
    def ntt(self):
        return self.engine.ntt

    def as_vertex(func):
        """Decorator to mark a function as a vertex in the computation graph.
        This decorator will record the start and end time of the function call,
        and will also record the number of times the function has been called.
        It will also check if the result of the function is a DataStruct, and if so, it will add the event information to the misc field of the DataStruct.

        Please note that this decorator will assume the function will inherit the misc from input to the returned restlt.

        Args:
            func (Callable): The function to be decorated.

        Returns:
            Callable: The decorated function that records event information.
        """

        # Get the signature of the function
        signature = inspect.signature(func)
        sig_names = list(signature.parameters.keys())
        # todo)) should I out signature on vertex?

        @wraps(func)
        def wrapper(self, *args, **kwargs):
            t0 = time()
            result = func(self, *args, **kwargs)
            t1 = time()
            # Record the start and end time of the function call
            self.event_counter[func.__name__] = (
                self.event_counter.get(func.__name__, 0) + 1
            )
            self.timeline.append((func.__name__, t0, t1))
            # The function should be a vertex in the graph
            vertex_name = func.__name__
            # check if anything in result is a DataStruct
            result_is_tuple = isinstance(result, tuple)
            if not result_is_tuple:
                result = (result,)
            for item in result:
                if isinstance(item, DataStruct):
                    if self.__class__.__name__ not in item.misc:
                        item.misc[self.__class__.__name__] = []
                    item.misc[self.__class__.__name__].append(
                        {
                            "event": vertex_name,
                            "start_time": t0,
                            "end_time": t1,
                        }
                    )
            if not result_is_tuple:
                result = result[0]
            return result

        return wrapper

    # def count_it(func):
    #     @wraps(func)
    #     def wrapper(self, *args, **kwargs):
    #         # Increment the class-level event counter
    #         self.event_counter[func.__name__] = (
    #             self.event_counter.get(func.__name__, 0) + 1
    #         )
    #         t0 = time()
    #         return_state = func(self, *args, **kwargs)
    #         # should I torch synchronize here?
    #         # torch.cuda.synchronize()  # todo test if this is necessary
    #         t1 = time()
    #         self.timeline.append(
    #             (func.__name__, t0, t1)
    #         )  # event_name, start_time, end_time
    #         return return_state

    #     return wrapper

    def reset(self):
        # events are distinguished by name, the counter is separated for each event
        self.event_counter = {}
        self.timeline = []

    def __init__(self, engine: CkksEngine):
        logger.warning(
            f"You are using the {self.__class__.__name__} instead of the original CkksEngine. This wrapper will record the time of each function call and the number of calls. HOWEVER, this will lead to CPU bound in some case and slow down the execution. Only use this for debugging purpose."
        )
        self.engine = engine
        self.reset()

    def plot_event_view(
        self,
        event_names: dict | None = None,
        limit: tuple | None = None,
        event_colors: dict[str, str] | None = None,
    ):
        """Plot a timeline visualization of the events that have occurred during the current session.

        Args:
            event_names (Optional[Dict], optional): A dictionary of event names to filter the timeline by. Defaults to None.
            limit (Optional[Tuple], optional): A tuple of (index_start, index_end) to limit the timeline by. Defaults to None.
            event_colors (Optional[Dict[str, str]], optional): A dictionary of event names to color values. Defaults to None.

        Returns:
            plotly.graph_objects.Figure: A timeline visualization of the events that have occurred during the current session.

        Note: to specify color, pass color dict in format of {event_name: color_value}. For example:
            event_colors = {
                "Event1": "red",
                "Event2": "blue",
                "Event3": "green",
            }
        """
        # Convert the list into a DataFrame
        timeline = self.timeline.copy()
        if event_names is not None:
            # filter the timeline by event names
            event_names = set(event_names)
            timeline = [event for event in timeline if event[0] in event_names]
        if limit is not None:
            # filter the timeline by time limit
            # limit should be a tuple of (index_start, index_end)
            timeline = timeline[limit[0] : limit[1]]

        df = pd.DataFrame(timeline, columns=["Event", "Start", "Finish"])
        # Convert the time scale from seconds to milliseconds
        df["Start"] = df["Start"] * 1000  # Convert Start to ms
        df["Finish"] = df["Finish"] * 1000  # Convert Finish to ms

        # Use a default color scale if no event_colors are provided
        if event_colors is None:
            color_scale = px.colors.qualitative.Plotly
            unique_events = df["Event"].unique()
            event_colors = {
                event: color_scale[i % len(color_scale)]
                for i, event in enumerate(unique_events)
            }

        # Create the timeline visualization with a consistent color map
        fig = px.timeline(
            df,
            x_start="Start",
            x_end="Finish",
            y="Event",
            title="Event Timeline",
            color="Event",
            color_discrete_map=event_colors,
        )

        # Add markers for each event, matching their assigned color
        fig.add_scatter(
            x=df["Start"],
            y=df["Event"],
            mode="markers",
            marker={
                "size": 8,
                "color": [event_colors[event] for event in df["Event"]],
            },
            name="Event Start",
        )

        fig.update_layout(
            xaxis_title="Time (ms)",  # Update the x-axis label
            hoverlabel={
                "namelength": -1
            },  # Ensure full names appear in hover labels
        )

        return fig

    @property
    @wraps(CkksEngine.pk)
    def pk(self):
        return self.engine.pk

    @property
    @wraps(CkksEngine.sk)
    def sk(self):
        return self.engine.sk

    @property
    @wraps(CkksEngine.rotk)
    def rotk(self):
        return self.engine.rotk

    @property
    @wraps(CkksEngine.gk)
    def gk(self):
        return self.engine.gk

    @property
    @wraps(CkksEngine.evk)
    def evk(self):
        return self.engine.evk

    @property
    def num_slots(self):
        return self.engine.num_slots

    @property
    def deviations(self):
        return self.engine.deviations

    @as_vertex
    @wraps(CkksEngine.rescale)
    def rescale(self, *args, **kwargs):
        return self.engine.rescale(*args, **kwargs)

    @as_vertex
    @wraps(CkksEngine.encode)
    def encode(self, *args, **kwargs):
        return self.engine.encode(*args, **kwargs)

    @as_vertex
    @wraps(CkksEngine.decode)
    def decode(self, *args, **kwargs):
        return self.engine.decode(*args, **kwargs)

    @as_vertex
    @wraps(CkksEngine.encodecrypt)
    def encodecrypt(self, *args, **kwargs):
        return self.engine.encodecrypt(*args, **kwargs)

    @as_vertex
    @wraps(CkksEngine.decryptcode)
    def decryptcode(self, *args, **kwargs):
        return self.engine.decryptcode(*args, **kwargs)

    @as_vertex
    @wraps(CkksEngine.pc_add)
    def pc_add(self, *args, **kwargs):
        return self.engine.pc_add(*args, **kwargs)

    @as_vertex
    @wraps(CkksEngine.pc_mult)
    def pc_mult(self, *args, **kwargs):
        return self.engine.pc_mult(*args, **kwargs)

    @as_vertex
    @wraps(CkksEngine.cc_add)
    def cc_add(self, *args, **kwargs):
        return self.engine.cc_add(*args, **kwargs)

    @as_vertex
    @wraps(CkksEngine.cc_add_triplet)
    def cc_add_triplet(self, *args, **kwargs):
        return self.engine.cc_add_triplet(*args, **kwargs)

    @as_vertex
    @wraps(CkksEngine.cc_mult)
    def cc_mult(self, *args, **kwargs):
        return self.engine.cc_mult(*args, **kwargs)

    @as_vertex
    @wraps(CkksEngine.mc_mult)
    def mc_mult(self, *args, **kwargs):
        return self.engine.mc_mult(*args, **kwargs)

    @as_vertex
    @wraps(CkksEngine.mc_add)
    def mc_add(self, *args, **kwargs):
        return self.engine.mc_add(*args, **kwargs)

    @as_vertex
    @wraps(CkksEngine.rotate_single)
    def rotate_single(self, *args, **kwargs):
        return self.engine.rotate_single(*args, **kwargs)

    @as_vertex
    @wraps(CkksEngine.rotate_galois)
    def rotate_galois(self, *args, **kwargs):
        return self.engine.rotate_galois(*args, **kwargs)

    @as_vertex
    @wraps(CkksEngine.rotate_offset)
    def rotate_offset(self, *args, **kwargs):
        return self.engine.rotate_offset(*args, **kwargs)

    @as_vertex
    @wraps(CkksEngine.sum)
    def sum(self, *args, **kwargs):
        return self.engine.sum(*args, **kwargs)

    @as_vertex
    @wraps(CkksEngine.level_up)
    def level_up(self, *args, **kwargs):
        return self.engine.level_up(*args, **kwargs)

    @as_vertex
    @wraps(CkksEngine.align_level)
    def align_level(self, *args, **kwargs):
        return self.engine.align_level(*args, **kwargs)

    @as_vertex
    @wraps(CkksEngine.relinearize)
    def relinearize(self, *args, **kwargs):
        return self.engine.relinearize(*args, **kwargs)

    def __repr__(self):
        return self.engine.__repr__()

    def __getattr__(self, name):
        # Check if the attribute exists in the original CkksEngine
        if hasattr(self.engine, name):
            return getattr(self.engine, name)
        else:
            raise AttributeError(f"{name} not found in CkksEngine.")
