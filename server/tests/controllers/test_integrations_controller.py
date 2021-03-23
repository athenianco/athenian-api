import pytest

from athenian.api.models.web import ContributorIdentity, MatchedIdentity
from athenian.api.serialization import FriendlyJson


async def test_match_identities_smoke(client, headers):
    body = {
        "account": 1,
        "identities": [
            {
                "names": ["Vadim", "Markovtsv"],
            },
            {
                "emails": ["eiso@athenian.co", "contact@eisokant.com"],
            },
            {
                "names": ["Denys Smyrnov"],
                "emails": ["denys@sourced.tech"],
            },
        ],
    }
    response = await client.request(
        method="POST", path="/v1/match/identities", headers=headers, json=body,
    )
    rbody = (await response.read()).decode("utf-8")
    assert response.status == 200, "Response body is : " + rbody
    model = [MatchedIdentity.from_dict(i) for i in FriendlyJson.loads(rbody)]
    for i in range(len(body["identities"])):
        assert model[i].from_ == ContributorIdentity.from_dict(body["identities"][i])
    assert model[0].to == "github.com/vmarkovtsev"
    assert model[0].confidence < 1
    assert model[1].to == "github.com/eiso"
    assert model[1].confidence == 1
    assert model[2].to == "github.com/dennwc"
    assert model[2].confidence < 1


@pytest.mark.parametrize("body, code", [
    ({"account": 1, "identities": [{}]}, 400),
    ({"account": 1, "identities": [{"emails": []}]}, 400),
    ({"account": 1, "identities": [{"names": []}]}, 400),
    ({"account": 1, "identities": [{"names": [], "emails": []}]}, 400),
    ({"account": 1, "identities": [{"emails": ["x@y.co"]}, {"emails": ["x@y.co"]}]}, 400),
    ({"account": 2, "identities": [{"emails": ["x@y.co"]}]}, 422),
    ({"account": 4, "identities": [{"emails": ["x@y.co"]}]}, 404),
])
async def test_match_identities_nasty_input(client, headers, body, code):
    response = await client.request(
        method="POST", path="/v1/match/identities", headers=headers, json=body,
    )
    rbody = (await response.read()).decode("utf-8")
    assert response.status == code, "Response body is : " + rbody
