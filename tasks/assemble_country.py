"""One saved-only sparse country reduction and local STAC assembly; no dispatch."""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import math
import os
from pathlib import Path
import time

import numpy as np
import pystac
import rasterio
from shapely.geometry import box, shape

from .native_checks import supplied_domain, generated_inputs, cog_native
from carbon.grid import retained_transform
from carbon.io import read_json, sha256, write_json
from .catalog import country_plan, resolve_window
from .completion import verify_completion
from .outputs import STATUS


def independent_native(native, task, package):
    """Read native layers, independently classify/reduce supplied physical hectares."""
    names = ['carbon','height','status','owner','physical-area-ha','WorldCover']
    values, masks = {}, {}
    for name in names:
        with rasterio.open(native/(name+'.tif')) as ds:
            if (ds.crs != rasterio.crs.CRS.from_user_input(task['grid']['crs'])
                    or ds.transform != retained_transform(task['grid'])
                    or ds.shape != tuple(task['grid']['retained_shape'])):
                raise ValueError('Native task grid differs: '+name)
            values[name], masks[name] = ds.read(1), ds.dataset_mask()>0
    w, status, owner = values['physical-area-ha'], values['status'], values['owner']
    carbon, height, cover = values['carbon'], values['height'], values['WorldCover']
    dom = supplied_domain(package, task)
    if (not np.array_equal(w, dom['weights']) or not np.array_equal(owner, dom['owner'])
            or not np.array_equal(cover, dom['worldcover']) or not np.isfinite(w).all() or (w<0).any()):
        raise ValueError('Native owner/physical-area/classification differs from supplied inputs')
    owned = w>0
    if any(not np.array_equal(masks[k],owned) for k in ['status','owner','physical-area-ha','WorldCover']):
        raise ValueError('Native ownership masks differ')
    if not np.isin(status[owned], [0,1,4,5,7]).all() or (status[~owned]!=6).any():
        raise ValueError('Unexpected exclusion/status category')
    assessed = np.isin(status,[4,7]) & owned
    if not np.array_equal(masks['carbon'], assessed) or not np.array_equal(status==5, owned & (cover==0)):
        raise ValueError('Carbon/unknown masks differ')
    if not np.array_equal(status==0, owned & (cover==80)):
        raise ValueError('Water status differs')
    modelled = (status==4)&owned
    expected = (3.744*height[modelled].astype(np.float64)**1.391).astype(np.float32)
    if ((not np.isfinite(carbon[assessed]).all()) or (carbon[assessed]<0).any()
            or not np.array_equal(carbon[modelled],expected) or np.any(carbon[status==7]!=0)):
        raise ValueError('Native carbon formula/valid-zero differs')

    def reduce(mask):
        classified = mask & assessed
        return dict(owned_ha=float(w[mask].sum()), known_land_ha=float(w[mask & dom['land']].sum()),
            water_ha=float(w[mask & (cover==80)].sum()), unknown_ha=float(w[mask & (cover==0)].sum()),
            modelled_ha=float(w[mask & (status==4)].sum()), assumed_zero_ha=float(w[mask & (status==7)].sum()),
            unavailable_ha=float(w[mask & dom['land'] & ~assessed].sum()), assessed_ha=float(w[classified].sum()),
            stock_tC=float(np.sum(w[classified]*carbon[classified].astype(np.float64),dtype=np.float64)),
            height_valid_ha=float(w[mask & dom['land'] & masks['height']].sum()))
    summary = reduce(owned)
    per_owner = {r['id']:reduce(owned & (owner==r['numeric_id'])) for r in task['inputs']['owners']}
    if abs(summary['owned_ha']-sum(summary[k] for k in ['water_ha','unknown_ha','unavailable_ha','assessed_ha']))>1e-6:
        raise ValueError('Independent category/area closure differs')
    return dict(summary=summary, owners=per_owner,
        modelled_zero_ha=float(w[modelled & (carbon==0)].sum()),
        formula_max_difference=0., grids_masks_owners_weights_exact=True)


def compare_summary(fresh, saved):
    differences = {k:fresh[k]-saved[k] for k in saved}
    if any(not math.isfinite(v) or abs(v) > (1. if k=='stock_tC' else 1e-6) for k,v in differences.items()):
        raise ValueError('Independent native/completion totals differ: '+str(differences))
    return differences


def assemble(catalog, country, year, results, output):
    selected = country_plan(catalog, country, year)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    inventory, items, summaries, owners, rectangles = [], [], [], {}, []
    shared_blocks, shared_checks = {}, []
    for window in selected['windows']:
        key = window['id']
        rect = box(*window['bounds'])
        if any(rect.intersection(other).area>1e-3 for other in rectangles):
            raise ValueError('Duplicate retained accounting area')
        rectangles.append(rect)
        path = results/key
        row = dict(window=key, status='unprepared')
        if key not in selected['jobs']:
            inventory.append(row)
            continue
        resolved = resolve_window(selected, key)
        if not path.exists():
            row['status']='not-executed'
        elif (path/'integration-failure.json').exists():
            row['status']='failed'
        elif not (path/'integration-completion.json').exists():
            row['status']='incomplete'
        else:
            completion = verify_completion(path, resolved['job'], resolved['binding'])
            task = read_json(resolved['binding']['task'])
            generated = read_json(path/'worker/completion.json')
            for block in task['inputs']['blocks']:
                name = Path(block['path']).name
                actual = path/'input/package'/block['path']
                if name in shared_blocks:
                    previous, digest = shared_blocks[name]
                    if digest != block['sha256']:
                        with rasterio.open(previous) as a,rasterio.open(actual) as b:
                            if ((a.crs,a.transform,a.shape,a.dtypes)!=(b.crs,b.transform,b.shape,b.dtypes)
                                    or not np.array_equal(a.read(),b.read()) or not np.array_equal(a.read_masks(),b.read_masks())):
                                raise ValueError('Shared global context block differs at window seam: '+name)
                    shared_checks.append(dict(block=name,values_masks_grid_exact=True,TIFF_bytes_equal=digest==block['sha256']))
                else:
                    shared_blocks[name] = (actual,block['sha256'])
            independent = independent_native(path/'worker/native',task,resolved['binding']['package'])
            input_checks = generated_inputs(path/'input/package',task,path/'worker',
                supplied_domain(resolved['binding']['package'],task), generated.get('completion_kind')=='no_model')
            delta = compare_summary(independent['summary'],generated['summary'])
            for owner, summary in independent['owners'].items():
                compare_summary(summary,generated['owners'][owner])
                owners.setdefault(owner,[]).append(summary)
            summaries.append(independent['summary'])
            row.update(status='complete', completion_kind=generated.get('completion_kind','model'),
                completion_sha256=sha256(path/'integration-completion.json'),
                independent=independent, completion_differences=delta, input_native_checks=input_checks)
            item = pystac.Item.from_file(str(path/completion['item']))
            from pystac.validation.local_validator import LocalValidator
            LocalValidator().item_validator().validate(item.to_dict())
            row['COG_native_checks'] = {name:cog_native(path/'worker/native'/(name+'.tif'),
                Path(item.assets[name].get_absolute_href())) for name in generated['files']}
            row['STAC_core_schema_valid'] = True
            # Rewrite only this verification row's immutable first publication.
            # It has not been written yet; the original Item/COGs remain unchanged.
            # New country Item metadata, not mutation of frozen window evidence.
            original = item.get_self_href()
            item.make_asset_hrefs_absolute()
            for link in item.links:
                href = link.get_href()
                if isinstance(href,str) and not '://' in href and not Path(href).is_absolute():
                    link.target = str((Path(original).parent/href).resolve())
            for rel in ['collection','root','parent','self']:
                item.remove_links(rel)
            item.properties['agc:scope']='retained window in country assembly; consult completeness inventory'
            item.add_link(pystac.Link('derived_from',original,media_type='application/geo+json',title='Unmodified window result'))
            items.append(item)
        inventory.append(row)
        write_json(output/(key+'-verification.json'),row)
    complete = all(r['status']=='complete' for r in inventory)
    totals = {k:math.fsum(r[k] for r in summaries) for k in summaries[0]} if summaries else None
    per_owner = {owner:{k:math.fsum(r[k] for r in rows) for k in rows[0]} for owner,rows in owners.items()}
    order = selected['plan'].get('owner_order')
    if complete and order is not None and set(order)!=set(per_owner):
        raise ValueError('Country owner coverage/order incomplete')
    if totals:
        compare_summary({k:math.fsum(r[k] for r in per_owner.values()) for k in totals}, totals)
    geometry_closure = {}
    if complete and selected['plan'].get('owners'):
        from carbon.physical_area import ground_area_ha
        for owner in selected['plan']['owners']:
            expected = ground_area_ha(shape(owner['geometry']),selected['plan']['grid']['crs'])
            difference = per_owner[owner['id']]['owned_ha']-expected
            tolerance = max(1e-7,expected*1e-8)  # Accepted physical-area partition tolerance.
            if abs(difference)>tolerance:
                raise ValueError('Country physical owner partition differs: '+owner['id'])
            geometry_closure[owner['id']] = dict(expected_geometry_ha=expected,difference_ha=difference,tolerance_ha=tolerance)
    country_record = dict(schema='agc-country-completion-v1', country=selected['entry']['country'],
        year=year, source_product='GeoMAD', source_version='0.4.0', ownership_grid=selected['entry'].get('ownership'), plan_sha256=sha256(selected['plan_file']),
        complete=complete, windows=inventory, summary=totals if complete else None,
        included_partial_summary=None if complete else totals, owners=per_owner,
        owner_order=order, native_reduction='float64 native densities * supplied physical hectares; independent sum, not overview/mean',
        shared_context_blocks_checked=shared_checks,unique_canonical_blocks=len(shared_blocks),physical_owner_closure=geometry_closure,
        no_extrapolation=True, statuses=STATUS, seconds=time.monotonic()-started,
        DEP_execution_qualified=False, permission='GADM externally unverified; authorised internal engineering')
    write_json(output/'country-summary.json',country_record)
    if items:
        bbox = [min(i.bbox[0] for i in items),min(i.bbox[1] for i in items),max(i.bbox[2] for i in items),max(i.bbox[3] for i in items)]
        collection = pystac.Collection(f'{country.lower()}-agc-{year}-local',
            'Indicative living aboveground vegetation carbon. '+('Complete planned country execution. ' if complete else 'INCOMPLETE planned execution. ')+
            'Missing imagery/unknown classification/water excluded; no extrapolation. Accuracy and sharing permissions unverified.',
            pystac.Extent(pystac.SpatialExtent([bbox]),pystac.TemporalExtent([[datetime(year,1,1,tzinfo=timezone.utc),datetime(year,12,31,23,59,59,tzinfo=timezone.utc)]])),license='proprietary')
        collection.extra_fields.update({'agc:complete':complete,'agc:country':selected['entry']['country'],
            'agc:source_version':'0.4.0','agc:plan_sha256':sha256(selected['plan_file'])})
        collection.set_self_href(str(output/'collection.json'))
        for item in items:
            collection.add_item(item)
            item.set_self_href(str(output/'items'/(item.id+'.json')))
            item.make_asset_hrefs_relative()
            for link in item.get_links('derived_from'):
                if '://' not in link.get_href():
                    link.target = os.path.relpath(link.get_href(),Path(item.get_self_href()).parent)
        collection.add_asset('accounting',pystac.Asset('country-summary.json',media_type='application/json',roles=['metadata']))
        collection.save(catalog_type=pystac.CatalogType.SELF_CONTAINED)
        loaded = pystac.Collection.from_file(str(output/'collection.json'))
        LocalValidator().collection_validator().validate(loaded.to_dict())
        for item in loaded.get_items():
            LocalValidator().item_validator().validate(item.to_dict())
            for asset in item.assets.values():
                if not Path(asset.get_absolute_href()).is_file():
                    raise ValueError('Country STAC asset does not resolve')
            for link in item.get_links('derived_from'):
                href = link.get_href()
                if '://' not in href and not (Path(item.get_self_href()).parent/href).resolve().is_file():
                    raise ValueError('Country STAC lineage reference does not resolve')
        if len(list(loaded.get_items())) != len(items):
            raise ValueError('Country STAC Item inventory differs')
    return country_record


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['catalog','results','output']:
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--country',required=True)
    p.add_argument('--year',type=int,required=True)
    a = p.parse_args()
    result = assemble(a.catalog.absolute(),a.country,a.year,a.results.absolute(),a.output.absolute())
    if not result['complete']:
        raise SystemExit('Incomplete country inventory retained; no complete-country total')


if __name__ == '__main__':
    main()
