import logging
import os
from tempfile import NamedTemporaryFile
from typing import Optional
from zipfile import ZipFile

from aiohttp import web
from aiohttp.abc import AbstractStreamWriter
from names_matcher import NamesMatcher
import numpy as np
import pandas as pd
from sqlalchemy import select

from athenian.api import metadata
from athenian.api.async_utils import gather
from athenian.api.balancing import weight
from athenian.api.internal.account import get_metadata_account_ids, get_user_account_status
from athenian.api.internal.features.everything import MineTopic, mine_everything
from athenian.api.internal.miners.github.contributors import load_organization_members
from athenian.api.internal.prefixer import Prefixer
from athenian.api.internal.settings import Settings
from athenian.api.models.state.models import UserAccount
from athenian.api.models.web import InvalidRequestError, MatchedIdentity, MatchIdentitiesRequest
from athenian.api.request import AthenianWebRequest
from athenian.api.response import ResponseError, model_response


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
                raise ResponseError(
                    InvalidRequestError(
                        detail="Emails of the identity must be unique: %s." % common_emails,
                        pointer=".identities[%d].emails" % i,
                    ),
                )
            match_by_email[i] = emails
            all_emails.update(emails)
        elif item.names:
            match_by_name[i] = set(item.names)
        else:
            raise ResponseError(
                InvalidRequestError(
                    detail="Identity must contain either `emails` or `names`.",
                    pointer=".identities[%d]" % i,
                ),
            )
    log.debug("to match by email: %d", len(match_by_email))
    meta_ids = await get_metadata_account_ids(model.account, request.sdb, request.cache)
    github_names, github_emails, github_prefixed_logins = await load_organization_members(
        model.account, meta_ids, request.mdb, request.sdb, log, request.cache,
    )
    inverted_github_emails = {}
    for node_id, emails in github_emails.items():
        for email in emails:
            if email not in inverted_github_emails:
                inverted_github_emails[email] = node_id
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
            matches[i].to = github_prefixed_logins[next(iter(node_ids))]
            matched_by_email += 1
    log.debug("matched by email: %d", matched_by_email)
    log.debug("to match by name: %d", len(match_by_name))
    match_users_keys = list(match_by_name)
    name_matches, confidences = NamesMatcher()(github_names.values(), match_by_name.values())
    matched_by_name = 0
    for github_user, match_index, confidence in zip(github_names, name_matches, confidences):
        if match_index >= 0 and confidence > 0:
            m = matches[match_users_keys[match_index]]
            m.to = github_prefixed_logins[github_user]
            m.confidence = confidence
            matched_by_name += 1
    log.debug("matched by name: %d", matched_by_name)
    log.info("matched %d / %d", sum(1 for m in matches if m.to is not None), len(matches))
    return model_response(matches)


class RemovingFileResponse(web.FileResponse):
    """Remove the served file when finished."""

    async def prepare(self, request: web.BaseRequest) -> Optional[AbstractStreamWriter]:
        """Serve the file response."""
        try:
            return await super().prepare(request)
        finally:
            os.remove(self._path)


def _df_to_parquet(df: pd.DataFrame, fout) -> None:
    zero = pd.Timestamp(0)
    for col in df:
        if df[col].dtype.type is np.datetime64:
            df[col] = df[col].dt.tz_localize(None)
        elif df[col].dtype.type is np.timedelta64:
            df[col] = zero + df[col]
    df.to_parquet(fout, engine="pyarrow", version="2.0")


_get_everything_formats = {
    "parquet": _df_to_parquet,
}


@weight(6.0)
async def get_everything(
    request: AthenianWebRequest,
    account: int = 0,
    format: str = "parquet",
) -> web.FileResponse:
    """Download all the data collected by Athenian for custom analysis."""
    if account == 0:
        rows = await request.sdb.fetch_all(
            select([UserAccount.account_id]).where(UserAccount.user_id == request.uid),
        )
        if len(rows) != 1:
            raise ResponseError(
                InvalidRequestError(
                    detail=(
                        "User belongs to %d accounts, must specify `account` URL query argument."
                    )
                    % len(rows),
                    pointer="account",
                ),
            )
        account = rows[0][0]
    else:
        await get_user_account_status(
            request.uid,
            account,
            request.sdb,
            request.mdb,
            request.user,
            request.app["slack"],
            request.cache,
        )
    meta_ids = await get_metadata_account_ids(account, request.sdb, request.cache)
    prefixer = await Prefixer.load(meta_ids, request.mdb, request.cache)
    settings = Settings.from_request(request, account)
    release_settings, logical_settings = await gather(
        settings.list_release_matches(),
        settings.list_logical_repositories(prefixer),
    )
    data = await mine_everything(
        set(MineTopic),
        release_settings,
        logical_settings,
        prefixer,
        account,
        meta_ids,
        request.sdb,
        request.mdb,
        request.pdb,
        request.rdb,
        request.cache,
    )
    serialize = _get_everything_formats[format]
    with NamedTemporaryFile(
        prefix=f"athenian_get_everything_{account}_", suffix=".zip", delete=False,
    ) as tmpf:
        with ZipFile(tmpf, "w") as zipf:
            for key, df_dict in data.items():
                for subkey, df in df_dict.items():
                    with zipf.open(f"{key.value}{subkey}.{format}", mode="w") as pf:
                        serialize(df, pf)
        return RemovingFileResponse(tmpf.name)
