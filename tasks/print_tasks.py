"""List an existing country/window plan; listing is not input or run admission."""

import argparse
import json
from pathlib import Path

from carbon.io import read_json
from .job import plan_windows
from .catalog import listed_tasks
from .job import resolve_job, verify_inputs
from .loading import local_pin
from carbon.io import check_file


def preparation_tasks(root,country,year,window=None):
    """Read-only progress from the normal producer checkpoint; no old folders."""
    state=read_json(root/'preparation-state.json');plan=read_json(root/'plan.json')
    check_file(root/'plan.json',state['contract']['plan_sha256'])
    if (state['contract']['country']['code'], state['contract']['year']) != (country, year):
        raise ValueError('Preparation progress belongs to another country/year')
    rows=[]
    for w in plan_windows(plan,country,year):
        if window is not None and w['id']!=window:
            continue
        row=dict(country=country,year=year,grid=plan['grid']['id'],window=w['id'],status='unprepared')
        if w['id'] in state['completed']:
            record=state['completed'][w['id']]
            verify_inputs(resolve_job(local_pin(root,record['job']),year,'0.4.0'))
            row['status']='configured'
        rows.append(row)
    if not rows:
        raise ValueError('No matching prepared-plan window')
    return rows


def tasks(plan, country, year, window=None):
    rows = [dict(country=country, grid=plan['grid']['id'], window=w['id'], year=year,
                 source_version='0.4.0', input_status='requires pinned job inputs')
            for w in plan_windows(plan, country, year) if window is None or w['id'] == window]
    if not rows:
        raise ValueError('No matching configured window')
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    choice = p.add_mutually_exclusive_group(required=True)
    choice.add_argument('--plan', type=Path)
    choice.add_argument('--catalog', type=Path)
    choice.add_argument('--preparation',type=Path,help='List verified preparation progress, including unprepared windows')
    p.add_argument('--country', required=True)
    p.add_argument('--year', type=int, required=True)
    p.add_argument('--window')
    p.add_argument('--source-version', default='0.4.0')
    p.add_argument('--pending-only', action='store_true', help='Exclude only verified completed jobs')
    p.add_argument('--ids', action='store_true', help='One window ID per line for the documented serial run loop')
    a = p.parse_args()
    if a.preparation:
        if a.pending_only:
            p.error('--pending-only is processing completion filtering; use --catalog')
        result=preparation_tasks(a.preparation,a.country,a.year,a.window)
    elif a.catalog:
        result = listed_tasks(a.catalog, a.country, a.year, a.window,
                              a.pending_only, a.source_version)
    else:
        if a.pending_only or a.source_version != '0.4.0':
            p.error('Completion filtering requires --catalog; only source0.4.0 is supported')
        result = tasks(read_json(a.plan), a.country, a.year, a.window)
    if a.ids:
        if any(r.get('window') is None for r in result):
            raise ValueError('Blocked plan has no executable window IDs')
        print('\n'.join(r['window'] for r in result))
    else:
        print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
