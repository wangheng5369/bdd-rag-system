from pytest_bdd import given, when, then, parsers


@given("已有可用集群")
def stepgivencluster(context):
    context["clusterid"] = "test-cluster-001"


@when("创建一个数据盘为100G的节点，规格为s7.large.2")
def stepcreatenodewith100gdatadisk(context):
    from cce_sdk.node import createnode
    result = createnode(
        clusterid=context["clusterid"],
        name="test-node-100g",
        flavor="s7.large.2",
        datavolumesize=100
    )
    context["nodeid"] = result["nodeid"]


@when("调用 resetnode 重置该节点")
def stepresetnode(context):
    from cce_sdk.node import resetnode
    resetnode(
        clusterid=context["clusterid"],
        nodeid=context["nodeid"],
        reinstallos=True
    )


@then('node的状态为"Active"')
def stepchecknodestatus(context):
    pass  # TODO: 缺少 API，无法实现


@then('node的规格为"s7.large.2"')
def stepchecknodeflavor(context):
    pass  # TODO: 缺少 API，无法实现


@then("node的cpu为2Core，内存为4G")
def stepchecknodecpumemory(context):
    pass  # TODO: 缺少 API，无法实现


@then('node的磁盘大小为100G，磁盘类型为"SAS"')
def stepchecknodedisk(context):
    pass  # TODO: 缺少 API，无法实现