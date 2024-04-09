read -p "USERNAME: " USER
read -p "PASSCODE: " -s PASSWORD

TOKEN=$(
 curl -s --location --request POST "https://obsidian.ccs.ornl.gov/token" \
             --header "Accept: application/json, text/plain, */*" \
         --header "Content-Type: application/x-www-form-urlencoded" \
                 --data-urlencode "username=$USER" \
                 --data-urlencode "password=$PASSWORD"
)

# TOKEN is JSON, extract the access_token prop
TOKEN=$(echo "$TOKEN" | jq -r ".access_token")
echo $TOKEN > .chathpc_token

curl https://obsidian.ccs.ornl.gov/chathpc/api/v1/models -H "Authorization: Bearer $TOKEN"

