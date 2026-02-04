
BOT_TOKEN="7583006976:AAGYVWlnGT_P-GKobSynnBDfSbKsxWqLUSo"
CHAT_ID="5397126439"

curl -s -X POST "https://api.telegram.org/bot$BOT_TOKEN/sendMessage" \
    -d chat_id=$CHAT_ID \
    -d text="Starting the processes with new initial contitions."

taskset -c 0-39 python Velocity_profile_fitting_chi2.py neidL2_20201221T200656.fits &
taskset -c 0-39 python Velocity_profile_fitting_chi2.py neidL2_20210101T172802.fits &
taskset -c 0-39 python Velocity_profile_fitting_chi2.py neidL2_20210101T202310.fits &
taskset -c 0-39 python Velocity_profile_fitting_chi2.py neidL2_20210204T195603.fits &
taskset -c 0-39 python Velocity_profile_fitting_chi2.py neidL2_20210301T163534.fits &
taskset -c 0-39 python Velocity_profile_fitting_chi2.py neidL2_20210301T210541.fits &
taskset -c 0-39 python Velocity_profile_fitting_chi2.py neidL2_20210401T164837.fits &
taskset -c 0-39 python Velocity_profile_fitting_chi2.py neidL2_20210403T200236.fits &
taskset -c 0-39 python Velocity_profile_fitting_chi2.py neidL2_20210503T175026.fits &
taskset -c 0-39 python Velocity_profile_fitting_chi2.py neidL2_20210504T184018.fits &
taskset -c 0-39 python Velocity_profile_fitting_chi2.py neidL2_20210611T173219.fits &
taskset -c 0-39 python Velocity_profile_fitting_chi2.py neidL2_20210711T194054.fits &
taskset -c 0-39 python Velocity_profile_fitting_chi2.py neidL2_20210730T174905.fits &
taskset -c 0-39 python Velocity_profile_fitting_chi2.py neidL2_20210824T211140.fits &
taskset -c 0-39 python Velocity_profile_fitting_chi2.py neidL2_20210916T211714.fits &
taskset -c 0-39 python Velocity_profile_fitting_chi2.py neidL2_20210923T211259.fits &
taskset -c 0-39 python Velocity_profile_fitting_chi2.py neidL2_20211001T172639.fits &
taskset -c 0-39 python Velocity_profile_fitting_chi2.py neidL2_20211011T212146.fits &
taskset -c 0-39 python Velocity_profile_fitting_chi2.py neidL2_20211111T220007.fits &
taskset -c 0-39 python Velocity_profile_fitting_chi2.py neidL2_20211202T170314.fits &
taskset -c 0-39 python Velocity_profile_fitting_chi2.py neidL2_20211219T165647.fits &
taskset -c 0-39 python Velocity_profile_fitting_chi2.py neidL2_20220101T180052.fits &
taskset -c 0-39 python Velocity_profile_fitting_chi2.py neidL2_20220116T205758.fits &
taskset -c 0-39 python Velocity_profile_fitting_chi2.py neidL2_20220212T200843.fits &
taskset -c 0-39 python Velocity_profile_fitting_chi2.py neidL2_20220312T182718.fits &
taskset -c 0-39 python Velocity_profile_fitting_chi2.py neidL2_20220403T202536.fits &
taskset -c 0-39 python Velocity_profile_fitting_chi2.py neidL2_20220424T164906.fits &
taskset -c 0-39 python Velocity_profile_fitting_chi2.py neidL2_20220515T171114.fits &
taskset -c 0-39 python Velocity_profile_fitting_chi2.py neidL2_20220530T172743.fits &
taskset -c 0-39 python Velocity_profile_fitting_chi2.py neidL2_20220601T173853.fits &
taskset -c 0-39 python Velocity_profile_fitting_chi2.py neidL2_20220606T173457.fits &

wait

curl -s -X POST "https://api.telegram.org/bot$BOT_TOKEN/sendMessage" \
    -d chat_id=$CHAT_ID \
    -d text="✅ All 31 processes have completed!. Chi2 results are saved"
