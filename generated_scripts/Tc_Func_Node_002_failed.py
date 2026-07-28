from pytest_bdd import given, when, then, parsers

============================================================
第一组场景: 硬编码 cluster-001 / nodepool-001
============================================================

@when('构造删除100G数据盘的节点请求，名称为"nodepool-100g-data"')
def stepbuilddeletereq1(context):
    context["deletereq"] = {"name": "nodepool-100g-data"}


@when('设置集群ID为"cluster-001"')
def stepsetclusterid1(context):
    context["deletereq"]["clusterid"] = "cluster-001"


@when('设置节点池ID为"nodepool-001"')
def stepsetnodepoolid1(context):
    context["deletereq"]["nodepoolid"] = "nodepool-001"


@when("基于构造的请求，删除100G数据盘的节点")
def stepcalldeletenodepool1(context):
    from cce_sdk.nodepool import deletenodepool
    req = context["deletereq"]
    result = deletenodepool(
        clusterid=req["clusterid"],
        nodepoolid=req["nodepoolid"]
    )
    context["deleteresult"] = result


============================================================
第二组场景: 参数化 clusterid / nodepoolid
============================================================

@when('构造删除100G数据盘的节点请求，名称为"100G-data-disk-nodepool"')
def stepbuilddeletereq2(context):
    context["deletereq"] = {"name": "100G-data-disk-nodepool"}


@when(parsers.parse('设置集群ID为"{clusterid}"'))
def stepsetclusterid2(context, clusterid):
    context["deletereq"]["clusterid"] = clusterid


@when(parsers.parse('设置节点池ID为"{nodepoolid}"'))
def stepsetnodepoolid2(context, nodepoolid):
    context["deletereq"]["nodepoolid"] = nodepoolid


@when("基于构造的请求，删除100G数据盘的节点")
def stepcalldeletenodepool2(context):
    from cce_sdk.nodepool import deletenodepool
    req = context["deletereq"]
    result = deletenodepool(
        clusterid=req["clusterid"],
        nodepoolid=req["nodepoolid"]
    )
    context["deleteresult"] = result


============================================================
第三组场景: 再次硬编码 cluster-001 / nodepool-001
============================================================

@when('构造删除100G数据盘的节点请求，名称为"nodepool-100g-data"')
def stepbuilddeletereq3(context):
    context["deletereq"] = {"name": "nodepool-100g-data"}


@when('设置集群ID为"cluster-001"')
def stepsetclusterid3(context):
    context["deletereq"]["clusterid"] = "cluster-001"


@when('设置节点池ID为"nodepool-001"')
def stepsetnodepoolid3(context):
    context["deletereq"]["nodepoolid"] = "nodepool-001"


@when("基于构造的请求，删除100G数据盘的节点")
def stepcalldeletenodepool3(context):
    from cce_sdk.nodepool import deletenodepool
    req = context["deletereq"]
    result = deletenodepool(
        clusterid=req["clusterid"],
        nodepoolid=req["nodepoolid"]
    )
    context["delete_result"] = result