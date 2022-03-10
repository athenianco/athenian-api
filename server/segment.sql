-- query to fetch all user activities from Segment
select t.timestamp, u.id, u.name, u.login, u.organizations::json->-1#>>'{}' as organization, u.accounts::json->-1 as account
from athenian_api.tracks t inner join athenian_api.users u
on t.user_id = u.id
order by timestamp;
