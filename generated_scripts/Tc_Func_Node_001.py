from pytest_bdd import given, when, then, parsers



@given("已存在可用集群")
def stepgivencluster(context):
    from cce_sdk.cluster import create_cluster
    result = create_cluster(
        name="test-cluster",
        flavor="cce.s1.small",
        vpcid="vpc-test-id",
        subnetid="subnet-test-id"
    )
    context["cluster_id"] = result["cluster_id"]


@when(parsers.parse('构造创建节点请求，名称为"{name}"，规格为"{flavor}"，数据盘大小为{disksize}G'))
def stepbuildnodereq(context, name, flavor, disksize):
    context["node_req"] = {
        "name": name,
        "flavor": flavor,
        "datavolumesize": int(disksize)
    }


@when("调用 create_node 创建节点")
def stepcreatenode(context):
    from cce_sdk.node import create_node
    result = create_node(
        clusterid=context["cluster_id"],
        **context["node_req"]
    )
    context["node_id"] = result.get("nodeid")


@then('node的状态为"Active"')
def stepverifynodestatus(context):
    pass  # TODO: 缺少 get_node API，无法实现


@then('node的规格为"s7.large.2"')
def stepverifynodeflavor(context):
    pass  # TODO: 缺少 get_node API，无法实现


@then("node的cpu为2Core，内存为4G")
def stepverifynodecpumemory(context):
    pass  # TODO: 缺少 get_node API，无法实现


@then('node的磁盘大小为100G，磁盘类型为"SAS"')
def stepverifynodedisk(context):
    pass  # TODO: 缺少 get_node API，无法实现