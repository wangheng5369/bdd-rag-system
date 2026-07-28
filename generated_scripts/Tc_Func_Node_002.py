from pytest_bdd import given, when, then, parsers

@when('构造删除100G数据盘的节点请求，名称为"100G数据盘节点池"')
def stepbuilddeletenodepoolreq(context):
    context["deletenodepoolreq"] = {"name": "100G数据盘节点池"}

@when('设置集群ID为"<clusterid>"')
def stepsetclusterid(context, clusterid):
    context["deletenodepoolreq"]["cluster_id"] = clusterid

@when('设置节点池ID为"<nodepoolid>"')
def stepsetnodepoolid(context, nodepoolid):
    context["deletenodepoolreq"]["nodepool_id"] = nodepoolid

@when("基于构造的请求，删除100G数据盘的节点")
def stepdeletenodepool(context):
    from cce_sdk.node import deletenodepool
    result = deletenodepool(
        clusterid=context["deletenodepoolreq"]["cluster_id"],
        nodepoolid=context["deletenodepoolreq"]["nodepool_id"]
    )
    context["deletenodepoolresult"] = result