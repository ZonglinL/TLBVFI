# rm -rf .git
git config --global user.name "ZonglinL"
git config --global user.email "zonglinlyu123123@gmail.com"
git remote set-url origin https://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@github.com/ZonglinL/TLBVFI
git init
git add . 
#git add README.md
#git add train/reward_control_gradient_straight.py 
git commit -m "intial commit"
git remote add origin https://github.com/ZonglinL/TLBVFI
git branch -M main
git push -u origin main