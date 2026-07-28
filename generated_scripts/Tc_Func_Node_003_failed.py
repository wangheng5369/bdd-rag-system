from pytest_bdd import given, when, then, parsers


@given("已存在可用集群")
def stepgivenclusterexists(context):
    from cce_sdk.cluster import create_cluster
    result = create_cluster(
        name="test-cluster-200g-node",
        flavor="cce.s1.small",
        vpcid="vpc-test-001",
        subnetid="subnet-test-001"
    )
    context["cluster_id"] = result["cluster_id"]

@when(parsers.parse("构造纳管节点请求，数据盘大小为{disk_size}"))
def step