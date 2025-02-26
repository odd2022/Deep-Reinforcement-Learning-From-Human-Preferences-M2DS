if __name__ == "__main__":
    import time
    import gymnasium as gym
    from trial import HumanPreferencesEnvWrapper

    env = HumanPreferencesEnvWrapper(gym.make("CartPole-v1"), segment_length=20)

    num_episodes = 10

    for episode in range(num_episodes):
        if episode == num_episodes // 2:
            env.switch_to_predicted_reward()
            print("🚀 Activation de la récompense prédite par le modèle !")

        obs, _ = env.reset()  # Gymnasium retourne un tuple (obs, info)
        done = False
        total_reward = 0

        while not done:
            action = env.action_space.sample()  # Prendre une action aléatoire
            obs, reward, done, info = env.step(action)
            total_reward += reward
            time.sleep(0.05)  # Pause pour mieux voir le déroulement

        print(f"Épisode {episode + 1} terminé avec un score de {total_reward}")

        # 🔹 Ajout du questionnement des préférences APRES chaque épisode
        pref = env.pref_interface.query_user()
        if pref:
            print("Préférence reçue, entraînement en cours...")
            s1, s2, preference = pref
            loss = env.reward_predictor.train_model(s1, s2, preference, env.optimizer, env.criterion)
            print(f"Loss: {loss}")

    print("✅ Entraînement terminé !")
