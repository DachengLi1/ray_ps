import ray
from pprint import pprint

# Import placement group APIs.
from ray.util.placement_group import (
    placement_group,
    placement_group_table,
    remove_placement_group
)

ray.init(num_gpus=2, num_cpus=4)

gpu_bundle = {"GPU": 2}
extra_resource_bundle = {"CPU": 2}

# Reserve bundles with strict pack strategy.
# It means Ray will reserve 2 "GPU" and 2 "extra_resource" on the same node (strict pack) within a Ray cluster.
# Using this placement group for scheduling actors or tasks will guarantee that they will
# be colocated on the same node.
pg = placement_group([gpu_bundle, extra_resource_bundle], strategy="SPREAD")

# Wait until placement group is created.
ray.get(pg.ready())

@ray.remote(num_gpus=1)
class GPUActor:
    def __init__(self):
        pass
    def execute(self):
        return 1 + 1

@ray.remote(num_cpus=1)
def extra_resource_task():
    import time
    # simulate long-running task.
    time.sleep(1)

# Create GPU actors on a gpu bundle.
gpu_actors = [GPUActor.options(
        placement_group=pg,
        # This is the index from the original list.
        # This index is set to -1 by default, which means any available bundle.
        placement_group_bundle_index=0) # Index of gpu_bundle is 0.
    .remote() for _ in range(2)]

# Create extra_resource actors on a extra_resource bundle.
extra_resource_actors = [extra_resource_task.options(
        placement_group=pg,
        # This is the index from the original list.
        # This index is set to -1 by default, which means any available bundle.
        placement_group_bundle_index=1) # Index of extra_resource_bundle is 1.
    .remote() for _ in range(2)]

ray.get(gpu_actors[0].execute.remote())
# This API is asynchronous.
remove_placement_group(pg)

# Wait until placement group is killed.
import time
time.sleep(1)
# Check the placement group has died.
pprint(placement_group_table(pg))

"""
{'bundles': {0: {'GPU': 2.0}, 1: {'extra_resource': 2.0}},
'name': 'unnamed_group',
'placement_group_id': '40816b6ad474a6942b0edb45809b39c3',
'state': 'REMOVED',
'strategy': 'STRICT_PACK'}
"""

ray.shutdown()

