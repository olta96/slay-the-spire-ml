package RecommenderSystem;

import basemod.BaseMod;
import basemod.interfaces.PostDungeonUpdateSubscriber;
import com.badlogic.gdx.Gdx;
import com.badlogic.gdx.Net;
import com.badlogic.gdx.net.HttpRequestBuilder;
import com.evacipated.cardcrawl.modthespire.lib.SpireInitializer;

import basemod.interfaces.PostBattleSubscriber;
import com.megacrit.cardcrawl.cards.AbstractCard;
import com.megacrit.cardcrawl.dungeons.AbstractDungeon;
import com.megacrit.cardcrawl.rewards.RewardItem;
import com.megacrit.cardcrawl.rooms.AbstractRoom;

import java.util.ArrayList;

@SpireInitializer
public class RecommenderSystemMod implements PostBattleSubscriber, PostDungeonUpdateSubscriber {

    private AbstractRoom rewardRoom = null;
    private final StateBuilder stateBuilder = new StateBuilder();

    public RecommenderSystemMod() {
        BaseMod.subscribe(this);
    }

    public static void initialize() {
        new RecommenderSystemMod();
    }

    @Override
    public void receivePostBattle(AbstractRoom abstractRoom) {
        rewardRoom = abstractRoom;
    }

    @Override
    public void receivePostDungeonUpdate() {
        if (
                AbstractDungeon.getCurrRoom() == rewardRoom &&
                AbstractDungeon.combatRewardScreen.rewards != null &&
                !AbstractDungeon.combatRewardScreen.rewards.isEmpty()
        ) {
            rewardRoom = null;
            for (RewardItem reward : AbstractDungeon.combatRewardScreen.rewards) {
                if (reward != null && reward.cards != null) {
                    State state = getState(reward.cards);
                    makeChoice(state);
                }
            }
        }
    }

    public State getState(ArrayList<AbstractCard> choices) {
        return stateBuilder
                .setChoices(choices)
                .setDeck(AbstractDungeon.player.masterDeck.group)
                .setRelics(AbstractDungeon.player.relics)
                .setFloor(AbstractDungeon.floorNum)
                .build();
    }

    public void makeChoice(State state) {
        String stateJSON = state.getJSON();

        HttpRequestBuilder requestBuilder = new HttpRequestBuilder();
        Net.HttpRequest request = requestBuilder.newRequest()
                .method("POST")
                .url("http://127.0.0.1:5000/make_choice")
                .header("Content-Type", "application/json")
                .header("Accept", "application/json")
                .build();
        request.setContent(stateJSON);
        Gdx.net.sendHttpRequest(request, new Net.HttpResponseListener() {
            @Override
            public void handleHttpResponse(Net.HttpResponse httpResponse) {
            }

            @Override
            public void failed(Throwable t) {
            }

            @Override
            public void cancelled() {
            }
        });
    }
}
