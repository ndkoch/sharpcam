import boto3
import time
import os
from botocore.exceptions import ClientError

ids = ["i-0531df58e59fca5e0"]
ec2 = boto3.client('ssm',region_name='us-east-2', aws_access_key_id='AKIAQ3DWRYZ7X54YJXKP',aws_secret_access_key='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')

# start the instance
start_ec2(ids)
time.sleep(10)

# wait for the instance to start and then run scp command
dns = ec2.describe_instances(InstanceIds=ids)['Reservations'][0]['Instances'][0]['PublicDnsName']
cmd = "ssh-keyscan -H %s >> ~/.ssh/known_hosts" % dns
os.system(cmd)
cmd = "scp -i /home/sharpcam/SharpCam/data/capstone_training_instance.pem /home/sharpcam/SharpCam/data/pre_img.zip ubuntu@%s:~/DeepVideoDeblurring/data/testing_real_all_nostab_nowarp/" % dns
os.system(cmd)

# wait for file to be completely uploaded before sending command to ec2 instance remotely
ssm_response = ec2.send_command(InstanceIds=ids, DocumentName='AWS-RunShellScript', Comment='begin video deblurring', Parameters={"commands":["python3 begin_deblur.py"]})
command_id = ssm_response['Command']['CommandId']
command_invocation_result = ec2.get_command_invocation(CommandId=command_id, InstanceId=ids[0])
while command_invocation_result['Status'] != 'Success':
  if command_invocation_result['Status'] == "Failed":
    print(command_invocation_result['StandardErrorContent'])
    break
  command_invocation_result = ec2.get_command_invocation(CommandId=command_id, InstanceId=ids[0])

# now the image should successfully be deblurred and all we need to do is zip the images and download them via scp
cmd = "scp -i /home/sharpcam/SharpCam/data/capstone_training_instance.pem ubuntu@%s:~/post_img.zip /home/sharpcam/SharpCam/data/pre_img.zip" % dns
os.system(cmd)
stop_ec2(ids)


# code from https://github.com/niftycode/aws-ec2-start-stop/blob/36a795d57802d82709fdd61f406880c6c0c5be52/start_stop_ec2.py#L132
def start_ec2(ids):
  try:
    ec2.start_instances(InstanceIds=ids, DryRun=True)
  except ClientError as e:
    if 'DryRunOperation' not in str(e):
        raise
  # Dry run succeeded, run start_instances without dryrun
  try:
    response = ec2.start_instances(InstanceIds=ids, DryRun=False)
  except ClientError as e:
    print(e)
def stop_ec2(ids):
    try:
      ec2.stop_instances(InstanceIds=ids, DryRun=True)
    except ClientError as e:
      if 'DryRunOperation' not in str(e):
          raise
    # Dry run succeeded, call stop_instances without dryrun
    try:
      response = ec2.stop_instances(InstanceIds=ids, DryRun=False)
    except ClientError as e:
      print(e)
