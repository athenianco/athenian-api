import logging

from aiohttp import web
from names_matcher import NamesMatcher

from athenian.api import metadata
from athenian.api.balancing import weight
from athenian.api.controllers.account import get_metadata_account_ids
from athenian.api.controllers.miners.github.contributors import load_organization_members
from athenian.api.models.metadata import PREFIXES
from athenian.api.models.web import InvalidRequestError, MatchedIdentity, MatchIdentitiesRequest
from athenian.api.request import AthenianWebRequest
from athenian.api.response import model_response, ResponseError


@weight(1.0)
async def match_identities(request: AthenianWebRequest, body: dict) -> web.Response:
    """Match provided people names/logins/emails to the account's GitHub organization members."""
    model = MatchIdentitiesRequest.from_dict(body)
    log = logging.getLogger("%s.match_identities" % metadata.__package__)
    match_by_email = {}
    all_emails = set()
    match_by_name = {}
    for i, item in enumerate(model.identities):
        if item.emails:
            if common_emails := (emails := set(item.emails)).intersection(all_emails):
                raise ResponseError(InvalidRequestError(
                    detail="Emails of the identity must be unique: %s." % common_emails,
                    pointer=".identities[%d].emails" % i))
            match_by_email[i] = emails
            all_emails.update(emails)
        elif item.names:
            match_by_name[i] = set(item.names)
        else:
            raise ResponseError(InvalidRequestError(
                detail="Identity must contain either `emails` or `names`.",
                pointer=".identities[%d]" % i))
    log.debug("to match by email: %d", len(match_by_email))
    meta_ids = await get_metadata_account_ids(model.account, request.sdb, request.cache)
    github_names, github_emails, github_logins = await load_organization_members(
        model.account, meta_ids, request.mdb, request.sdb, log)
    inverted_github_emails = {}
    for node_id, emails in github_emails.items():
        for email in emails:
            if email not in inverted_github_emails:
                inverted_github_emails[email] = node_id
    prefix = PREFIXES["github"]
    matches = [MatchedIdentity(from_=item, to=None, confidence=1) for item in model.identities]
    matched_by_email = 0
    for i, emails in match_by_email.items():
        node_ids = {inverted_github_emails.get(email) for email in emails}
        if None in node_ids:
            node_ids.remove(None)
        if not node_ids or len(node_ids) > 1:
            if names := model.identities[i].names:
                match_by_name[i] = set(names)
        else:
            matches[i].to = prefix + github_logins[next(iter(node_ids))]
            matched_by_email += 1
    log.debug("matched by email: %d", matched_by_email)
    log.debug("to match by name: %d", len(match_by_name))
    match_users_keys = list(match_by_name)
    name_matches, confidences = NamesMatcher()(github_names.values(), match_by_name.values())
    matched_by_name = 0
    for github_user, match_index, confidence in zip(github_names, name_matches, confidences):
        if match_index >= 0 and confidence > 0:
            m = matches[match_users_keys[match_index]]
            m.to = prefix + github_logins[github_user]
            m.confidence = confidence
            matched_by_name += 1
    log.debug("matched by name: %d", matched_by_name)
    log.info("matched %d / %d", sum(1 for m in matches if m.to is not None), len(matches))
    return model_response(matches)
