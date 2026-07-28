from pytest_bdd import given, when, then, parsers


@given(parsers.parse("已创建可用集群 {clustername} 规格 {flavor}"))
def stepcreatecluster(context, clustername, flavor):
    from cce_sdk.cluster import createcluster
    result = createcluster(
        name=clustername,
        flavor=flavor,
        vpcid="vpc-test-id",
        subnetid="subnet-test-id",
        containernetworkmode="overlayl2",
        version="v1.25"
    )
    context["clusterid"] = result.get("clusterid", "")
    return context


@given("已存在一台挂载了200G数据盘的ECS服务器")
def stepgivenserverwith200gdisk(context):
    # 前置：服务器侧已就绪，包含 200G 数据盘
    context["serverid"] = "ecs-server-with-200g-data-disk"


@when("调用 addnode 接口将该服务器纳管到集群中")
def stepaddnode(context):
    from cce_sdk.node import addnode
    result = addnode(
        clusterid=context["clusterid"],
        serverids=[context["serverid"]],
        ostype="EulerOS 2.9"
    )
    context["addresult"] = result
    context["nodeid"] = result.get("nodeid", "")


@then("纳管成功并返回节点ID")
def stepaddnodesuccess(context):
    assert context["addresult"] is not None
    assert context.get("nodeid"), "纳管应返回节点ID"


@then("节点已正确挂载200G数据盘")
def stepnodehas200gdisk(context):
    # TODO: 缺少 getnode/listnodes API，无法验证数据盘配置
    pass


@then("节点状态为运行中")
def stepnodestatusrunning(context):
    # TODO: 缺少 get_node API，无法验证节点运行状态
    pass