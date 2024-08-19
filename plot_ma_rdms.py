"""Title:Plot RDM for Multiple Arrangements tasks"""
"""ingredients: Model, Stimuli, Neuroimaging, Behavior"""

from __future__ import annotations
from typing import TYPE_CHECKING
import rsatoolbox, numpy
from matplotlib import pyplot
if TYPE_CHECKING:
    from jobcontext import JobContext


def main(job: JobContext):
    n = len(job.data)
    if n == 0:
        job.log('Zero participants')
        return
    for p, (pnick, pdata) in enumerate(job.data.items(), start=1):
        job.log(f'working on participant {p}/{n}', p/n)
        if 'tasks' not in pdata:
            job.log('missing tasks')
            continue
        for _, task in enumerate(pdata['tasks']):
            task_meta = task.get('task', {})
            if task_meta.get('task_type') == 'multiarrange':
                task_stimuli = [s['name'].split('.')[0] for s in task['stimuli']]
                rdms = rsatoolbox.rdm.rdms.RDMs(
                    numpy.atleast_2d(task['rdm']),
                    dissimilarity_measure='euclidean',
                    rdm_descriptors=dict(participation=[pnick]),
                    pattern_descriptors=dict(conds=task_stimuli),
                )
                pyplot.close('all')
                fig, _, _ = rsatoolbox.vis.rdm_plot.show_rdm(rdms)
                fpath = job.outputPath.joinpath(f'{pnick}.png')
                pyplot.savefig(fpath)
                job.addFile(fpath)
                break
